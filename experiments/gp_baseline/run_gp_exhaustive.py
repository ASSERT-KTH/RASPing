import argparse
import csv
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

import parso

from experiments.mutation.load_mutations import load_mutation
from experiments.gp_baseline.gp_core import (
    canonicalize_for_dataset,
    compute_max_length,
    discover_mutation_sites,
    evaluate_source,
    evaluate_source_test,
    hash_source,
)
from src.functions import getAcceptedNamesAndInput, load_dataset


logger = logging.getLogger(__name__)


CSV_PATH = (
    Path(__file__).parent.parent
    / "train_mutations"
    / "analysis_outputs"
    / "all_test_accuracies.csv"
)


def load_job_ids_from_csv(csv_path: Path) -> Set[str]:
    """Load job_ids from the CSV file."""
    job_ids = set()
    if not csv_path.exists():
        logging.warning(f"CSV file not found: {csv_path}")
        return job_ids
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_ids.add(row["job_id"])
    return job_ids


# -----------------------------
# Deterministic neighbor generation
# -----------------------------


def _apply_mutation_by_index(
    source_code: str, site_index: int, position_index: int
) -> Optional[str]:
    """
    Apply one mutation deterministically by choosing the Nth discovered site and the Mth
    mutation position for that site. Returns mutated source code or None if invalid.
    """
    try:
        tree = parso.parse(source_code)
    except Exception:
        return None

    sites = discover_mutation_sites(tree)
    if site_index < 0 or site_index >= len(sites):
        return None

    site = sites[site_index]
    if position_index not in site.indices:
        return None

    operator = site.operator_cls()
    try:
        mutated_node = operator.mutate(site.node, position_index)
    except Exception:
        return None

    if mutated_node is not site.node:
        parent = getattr(site.node, "parent", None)
        if (
            parent is not None
            and hasattr(parent, "children")
            and parent.children is not None
        ):
            for i, child in enumerate(parent.children):
                if child is site.node:
                    parent.children[i] = mutated_node
                    break

    try:
        mutated_source = tree.get_code()
    except Exception:
        return None

    if mutated_source == source_code:
        return None
    return mutated_source


def _all_single_step_neighbors(source_code: str) -> List[str]:
    """Enumerate all unique single-step mutations of source_code deterministically."""
    try:
        tree = parso.parse(source_code)
    except Exception:
        return []

    sites = discover_mutation_sites(tree)
    neighbors: List[str] = []

    for site_idx, site in enumerate(sites):
        for pos_idx in site.indices:
            mutated = _apply_mutation_by_index(source_code, site_idx, pos_idx)
            if not mutated:
                continue
            neighbors.append(mutated)
    return neighbors


# -----------------------------
# Exhaustive search core
# -----------------------------


@dataclass
class ExhaustiveResult:
    status: str
    num_programs: int
    num_duplicates: int
    num_compile_failures: int
    train_acc: float
    test_acc: float
    best_source: str
    depth_history: List[Dict[str, Any]]


def run_exhaustive_for_bug(
    mutation_entry: Dict[str, Any],
    data_dir: Path,
    accepted_inputs: Sequence[Any],
    max_depth: int = 2,
    budget: int = 10000,
    log_every: int = 50,
) -> Dict[str, Any]:
    import time

    program_name_raw = mutation_entry["program_name"]
    dataset_name = canonicalize_for_dataset(program_name_raw)
    base_source = mutation_entry["program_source_after"]

    # Load datasets
    train_data = load_dataset(data_dir, dataset_name, "train")
    val_data = load_dataset(data_dir, dataset_name, "val")
    test_data = load_dataset(data_dir, dataset_name, "test")
    fitness_data = train_data + val_data
    max_length = compute_max_length(data_dir, dataset_name)

    cache: Dict[str, Tuple[float, float]] = {}
    num_programs = 0
    num_duplicates = 0
    num_compile_failures = 0

    best_train = -1.0
    best_test = -1.0
    best_source = base_source

    program_history: List[Dict[str, Any]] = []

    def evaluate_candidate(
        src: str, *, depth_mark: int
    ) -> Tuple[str, Optional[Tuple[float, float]], str]:
        """Evaluate a candidate and return (status, (train,test) or None, hash)."""
        nonlocal num_programs, num_compile_failures, num_duplicates
        h = hash_source(src)
        # Count every attempt toward budget, mirroring baseline semantics
        num_programs += 1
        if h in cache:
            num_duplicates += 1
            return "duplicate", cache[h], h
        try:
            train_acc = evaluate_source(
                src, dataset_name, max_length, accepted_inputs, fitness_data
            )
            test_acc = evaluate_source_test(
                src, dataset_name, max_length, accepted_inputs, test_data
            )
        except Exception as e:
            num_compile_failures += 1
            logger.debug(f"Compile/eval failed for candidate {h}: {e}")
            return "compile_fail", None, h
        cache[h] = (train_acc, test_acc)
        return "ok", (train_acc, test_acc), h

    # Evaluate base
    start_time = time.time()
    status, base_eval, base_hash = evaluate_candidate(base_source, depth_mark=0)
    if status != "compile_fail" and base_eval is not None:
        best_train, best_test = base_eval
        logger.info(
            f"Seeded base candidate: train={best_train:.4f} test={best_test:.4f}"
        )
    program_history.append(
        {
            "step": num_programs,
            "depth": 0,
            "status": status,
            "train_acc": None if base_eval is None else base_eval[0],
            "test_acc": None if base_eval is None else base_eval[1],
            "best_train_acc": best_train,
            "best_test_acc": best_test,
        }
    )

    depth = 0
    frontier: List[str] = [base_source]

    if best_train >= 1.0:
        status = "success_early"
        end_time = time.time()
        result = {
            "status": status,
            "num_programs": num_programs,
            "num_duplicates": num_duplicates,
            "num_compile_failures": num_compile_failures,
            "train_acc": best_train,
            "test_acc": best_test,
            "best_source": best_source,
            "program_history": program_history,
            "job_id": mutation_entry.get("job_id"),
            "program_name": mutation_entry.get("program_name"),
            "duration_s": end_time - start_time,
        }
        return result

    # BFS over depths
    while depth < max_depth and frontier and num_programs < budget:
        next_frontier: List[str] = []
        # Expand all neighbors of current frontier
        for src in frontier:
            neighbors = _all_single_step_neighbors(src)
            for msrc in neighbors:
                next_frontier.append(msrc)

        # Evaluate next frontier
        depth += 1
        new_frontier: List[str] = []
        for idx, src in enumerate(next_frontier):
            if num_programs >= budget:
                break
            status_i, eval_res, cand_hash = evaluate_candidate(src, depth_mark=depth)
            train_acc = eval_res[0] if eval_res is not None else None
            test_acc = eval_res[1] if eval_res is not None else None
            if status_i == "ok" and eval_res is not None:
                new_frontier.append(src)
                if train_acc > best_train or (
                    train_acc == best_train and test_acc > best_test
                ):
                    best_train = train_acc
                    best_test = test_acc
                    best_source = src
                    logger.info(
                        f"New best: train={best_train:.4f} test={best_test:.4f}"
                    )
            program_history.append(
                {
                    "step": num_programs,
                    "depth": depth,
                    "status": status_i,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "best_train_acc": best_train,
                    "best_test_acc": best_test,
                }
            )
            if log_every and num_programs % max(1, log_every) == 0:
                logger.info(
                    f"Progress: depth={depth} programs={num_programs}/{budget} dup={num_duplicates} fails={num_compile_failures} best_train={best_train:.4f} best_test={best_test:.4f}"
                )
            if best_train >= 1.0:
                break

        frontier = new_frontier

        if best_train >= 1.0 or num_programs >= budget:
            break

    end_time = time.time()
    status = "success_early" if best_train >= 1.0 else "complete"
    result = {
        "status": status,
        "num_programs": num_programs,
        "num_duplicates": num_duplicates,
        "num_compile_failures": num_compile_failures,
        "train_acc": best_train,
        "test_acc": best_test,
        "best_source": best_source,
        "program_history": program_history,
        "job_id": mutation_entry.get("job_id"),
        "program_name": mutation_entry.get("program_name"),
        "duration_s": end_time - start_time,
    }
    return result


# -----------------------------
# CLI runner mirroring baseline
# -----------------------------


def _run_single_job(
    entry: Dict[str, Any], data_dir: Path, accepted_inputs: List[Any], args
) -> Tuple[str, Dict[str, Any]]:
    dataset_name = canonicalize_for_dataset(entry["program_name"])
    # Get max_depth from mutation_order: if mutation_order is 1, max_depth should be 1
    mutation_order = entry.get("mutation_order")
    logger.info(f"Mutation order: {mutation_order}")
    max_depth = mutation_order
    result = run_exhaustive_for_bug(
        mutation_entry=entry,
        data_dir=data_dir,
        accepted_inputs=accepted_inputs,
        max_depth=max_depth,
        budget=args.budget,
        log_every=args.log_every,
    )
    return dataset_name, result


def main():
    parser = argparse.ArgumentParser(
        description="Run exhaustive AST mutation search to repair buggy RASP programs"
    )
    parser.add_argument(
        "--mutation-path",
        type=str,
        required=True,
        help="Path to aggregated_mutations.json",
    )
    parser.add_argument(
        "--program-name",
        type=str,
        default=None,
        help="Program name to filter (e.g., reverse, sort, hist, most-freq, shuffle_dyck1, shuffle_dyck2)",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Specific job_id to run (overrides --n-jobs if provided)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of jobs to run for the program name (ignored if --job-id is set)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing program datasets",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10000,
        help="Attempt budget (candidate eval attempts) per bug",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (unused; kept for interface consistency)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/gp_baseline/results",
        help="Directory to write per-bug JSON results",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-every", type=int, default=50, help="Log progress every N attempts"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of skipping",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    # Load job IDs from CSV
    csv_job_ids = load_job_ids_from_csv(CSV_PATH)
    logging.info(f"Loaded {len(csv_job_ids)} job IDs from CSV")

    # Load matching mutation entries
    mutations: Dict[str, Dict[str, Any]] = load_mutation(
        mutation_path=args.mutation_path,
        program_name=args.program_name if args.program_name else None,
        job_id=args.job_id if args.job_id else None,
    )

    # Filter mutations to only include those in CSV
    mutations = {
        job_id: entry for job_id, entry in mutations.items() if job_id in csv_job_ids
    }
    logging.info(f"Filtered to {len(mutations)} mutations that appear in CSV")

    if args.job_id:
        to_run: List[Dict[str, Any]] = (
            [mutations[args.job_id]] if args.job_id in mutations else []
        )
    elif args.n_jobs:
        to_run = [row for _, row in list(mutations.items())[: max(args.n_jobs, 0)]]
    else:
        to_run = [row for _, row in list(mutations.items())]

    if not to_run:
        print("No matching buggy mutations found with the given filters.")
        return

    accepted_inputs_map = getAcceptedNamesAndInput()

    # Filter out jobs that already have results (resume) unless overwrite
    filtered: List[Dict[str, Any]] = []
    for entry in to_run:
        program_name_raw = entry["program_name"]
        dataset_name = canonicalize_for_dataset(program_name_raw)
        job_id = entry.get("job_id", "unknown")
        out_path = output_dir / f"{dataset_name}_{job_id}.json"
        if out_path.exists() and not args.overwrite:
            logging.info(f"Skipping existing result: {out_path}")
            continue
        filtered.append(entry)

    if not filtered:
        logging.info("All selected jobs already completed. Nothing to do.")
        return

    workers = max(1, args.workers)
    logging.info(f"Launching {len(filtered)} jobs with workers={workers}")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for entry in filtered:
            program_name_raw = entry["program_name"]
            dataset_name = canonicalize_for_dataset(program_name_raw)
            if dataset_name not in accepted_inputs_map:
                logging.error(f"Unsupported program_name: {program_name_raw}")
                continue
            accepted_inputs = accepted_inputs_map[dataset_name]
            futures[
                ex.submit(_run_single_job, entry, data_dir, accepted_inputs, args)
            ] = entry
        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                dataset_name, result = fut.result()
            except Exception as e:
                logging.error(f"Job {entry.get('job_id')} failed: {e}")
                continue
            job_id = entry.get("job_id", "unknown")
            out_path = output_dir / f"{dataset_name}_{job_id}.json"
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp_path, "w") as f:
                json.dump(result, f)
            tmp_path.replace(out_path)
            logging.info(f"Wrote result: {out_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
import logging
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.mutation.load_mutations import load_mutation
from experiments.gp_baseline.gp_core import run_gp_for_bug, canonicalize_for_dataset
from src.functions import getAcceptedNamesAndInput


CSV_PATH = Path(__file__).parent.parent / "train_mutations" / "analysis_outputs" / "all_test_accuracies.csv"


def load_job_ids_from_csv(csv_path: Path) -> Set[str]:
    """Load job_ids from the CSV file."""
    job_ids = set()
    if not csv_path.exists():
        logging.warning(f"CSV file not found: {csv_path}")
        return job_ids
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_ids.add(row['job_id'])
    return job_ids


def _run_single_job(
    entry: Dict[str, Any],
    data_dir: Path,
    accepted_inputs: List[Any],
    budget: int,
    population_size: int,
    tournament_k: int,
    seed: int,
    log_every: int,
    eval_timeout: float,
) -> Tuple[str, Dict[str, Any]]:
    dataset_name = canonicalize_for_dataset(entry["program_name"])
    result = run_gp_for_bug(
        mutation_entry=entry,
        data_dir=data_dir,
        accepted_inputs=accepted_inputs,
        budget=budget,
        population_size=population_size,
        tournament_k=tournament_k,
        seed=seed,
        log_every=log_every,
        eval_timeout=eval_timeout,
    )
    return dataset_name, result


def main():
    parser = argparse.ArgumentParser(description="Run GP baseline to repair buggy RASP programs")
    parser.add_argument("--mutation-path", type=str, required=True, help="Path to aggregated_mutations.json")
    parser.add_argument("--program-name", type=str, default=None, help="Program name to filter (e.g., reverse, sort, hist, most-freq, shuffle_dyck1, shuffle_dyck2)")
    parser.add_argument("--job-id", type=str, default=None, help="Specific job_id to run (overrides --n-jobs if provided)")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of jobs to run for the program name (ignored if --job-id is set)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing program datasets")
    parser.add_argument("--budget", type=int, default=500, help="Max successful compile+eval candidates per bug")
    parser.add_argument("--population-size", type=int, default=16, help="Population size μ and offspring λ per generation")
    parser.add_argument("--tournament-k", type=int, default=3, help="Tournament size for parent selection")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output-dir", type=str, default="experiments/gp_baseline/results", help="Directory to write per-bug JSON results")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("--log-every", type=int, default=25, help="Log progress every N successful evals")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--eval-timeout", type=float, default=30.0, help="Timeout in seconds for evaluating all samples in a dataset (default: 30.0)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results instead of skipping")

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level), format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')

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
    mutations = {job_id: entry for job_id, entry in mutations.items() if job_id in csv_job_ids}
    logging.info(f"Filtered to {len(mutations)} mutations that appear in CSV")

    if args.job_id:
        # Single job expected
        to_run: List[Dict[str, Any]] = [mutations[args.job_id]] if args.job_id in mutations else []
    elif args.n_jobs:
        # Take first N jobs by insertion order
        to_run = [row for _, row in list(mutations.items())[: max(args.n_jobs, 0)]]
    else:
        to_run = [row for _, row in list(mutations.items())]

    if not to_run:
        print("No matching buggy mutations found with the given filters.")
        return

    # Prepare accepted inputs map
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

    # Execute jobs in parallel
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
            # Extract args values to avoid pickling issues with argparse.Namespace
            futures[ex.submit(
                _run_single_job,
                entry,
                data_dir,
                accepted_inputs,
                args.budget,
                args.population_size,
                args.tournament_k,
                args.seed,
                args.log_every,
                args.eval_timeout,
            )] = entry
        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                dataset_name, result = fut.result()
            except Exception as e:
                job_id = entry.get('job_id', 'unknown')
                logging.error(f"Job {job_id} failed: {e}")
                logging.debug(f"Job {job_id} traceback: {traceback.format_exc()}")
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



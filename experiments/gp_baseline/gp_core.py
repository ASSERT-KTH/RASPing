import hashlib
import random
import time
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import parso
import logging

# Reuse existing utilities from the repo
from experiments.mutation.load_mutations import create_module_from_source
from experiments.mutation.operators.operators.provider import Provider
from src.functions import load_dataset
from tracr.rasp import rasp


logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


class TimeoutHandler:
    """Context manager for timeout using signals (Unix only)."""
    
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.old_handler = None
        
    def __enter__(self):
        if sys.platform == 'win32':
            # Windows doesn't support SIGALRM, fall back to no timeout
            logger.warning("Signal-based timeout not supported on Windows, timeout disabled")
            return self
        self.old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.platform != 'win32':
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.old_handler)
        return False
    
    def _timeout_handler(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.timeout}s")


# -----------------------------
# Name canonicalization helpers
# -----------------------------


def canonicalize_for_dataset(program_name: str) -> str:
    if program_name == "most_freq":
        return "most-freq"
    if program_name == "shuffle_dyck":
        return "shuffle_dyck1"
    return program_name


def canonicalize_for_factory(program_name: str) -> str:
    # Factory/source function names used inside mutated source modules
    if program_name == "most-freq":
        return "most_freq"
    if program_name == "shuffle_dyck1":
        return "shuffle_dyck"
    return program_name


# -----------------------------
# Data utilities
# -----------------------------


def compute_max_length(data_dir: Path, dataset_program_name: str) -> int:
    train = load_dataset(data_dir, dataset_program_name, "train")
    val = load_dataset(data_dir, dataset_program_name, "val")
    test = load_dataset(data_dir, dataset_program_name, "test")
    return max(max(len(inp) for inp, _ in train + val + test), 1)


# -----------------------------
# Model building and evaluation
# -----------------------------


def build_program_from_source(
    source_code: str,
    program_name: str,
    max_length: int,
    accepted_inputs: Sequence[Any],
    timeout: float = 30.0,
) -> Any:
    """
    Build a RASP SOp from mutated program source, following the same conventions
    used in experiments/mutation/load_mutations.py, but without compiling to a model.
    
    Args:
        timeout: Maximum time (in seconds) allowed for building the program
    """
    # Use signal-based timeout to interrupt blocking operations
    try:
        with TimeoutHandler(timeout):
            module_name = (
                f"gp_candidate_{hashlib.sha1(source_code.encode('utf-8')).hexdigest()[:10]}"
            )
            module = create_module_from_source(source_code, module_name)

            factory_name = canonicalize_for_factory(program_name)

            # Build the RASP program using the expected factory function in the module
            if factory_name == "hist":
                program = module.make_hist()
            elif factory_name == "sort":
                # min_key required by make_sort
                program = module.make_sort(
                    rasp.tokens,
                    rasp.tokens,
                    max_seq_len=max_length,
                    min_key=min(accepted_inputs),
                )
            elif factory_name == "reverse":
                program = module.make_reverse(rasp.tokens)
            elif factory_name == "most_freq":
                program = module.make_sort_freq(max_length)
            elif factory_name == "shuffle_dyck":
                program = module.make_shuffle_dyck(["()"])
            elif factory_name == "shuffle_dyck2":
                program = module.make_shuffle_dyck2()
            else:
                raise NotImplementedError(f"Program factory not implemented for {program_name}")

            return program
    except TimeoutError:
        logger.warning(f"Program building timeout after {timeout}s")
        raise TimeoutError(f"Program building exceeded {timeout}s timeout")
    except Exception as e:
        logger.debug(f"Program building failed: {e}")
        raise


def evaluate_program_on_split(
    program: Any, data: List[Tuple[List[Any], List[Any]]], timeout: float = 30.0
) -> float:
    """
    Evaluate a RASP SOp directly on input sequences.

    We ignore the first position (BOS-like) when computing sequence-level exact
    match, comparing positions 1..N of the output to the targets.

    Args:
        program: The RASP program to evaluate
        data: List of (input_seq, target_seq) tuples
        timeout: Maximum time (in seconds) allowed for evaluating all input sequences
    """
    if not data:
        return 0.0

    # Use signal-based timeout to interrupt blocking operations
    logger.debug(f"Starting evaluation of {len(data)} samples with {timeout}s timeout")
    start_time = time.time()
    
    try:
        with TimeoutHandler(timeout):
            correct = 0
            total = 0
            for idx, (input_seq, target_seq) in enumerate(data):
                # Log progress every 10% of samples
                if len(data) > 10 and idx % max(1, len(data) // 10) == 0:
                    elapsed = time.time() - start_time
                    logger.debug(f"Evaluating sample {idx}/{len(data)} (elapsed: {elapsed:.2f}s)")
                
                # Remove BOS-like token at position 0
                x = input_seq[1:]
                y = target_seq[1:]
                try:
                    pred = program(x)
                except Exception:
                    # Treat runtime errors as incorrect
                    pred = None
                total += 1
                if pred is not None and list(pred) == list(y):
                    correct += 1
            
            elapsed = time.time() - start_time
            logger.debug(f"Completed evaluation of {total} samples in {elapsed:.2f}s")
            result = correct / total if total > 0 else 0.0
            logger.debug(f"Evaluation completed in {elapsed:.2f}s, accuracy: {result:.4f}")
            return result
    except TimeoutError:
        # Timeout occurred - treat entire evaluation as failed (return 0.0)
        elapsed = time.time() - start_time
        logger.warning(
            f"Evaluation timeout after {elapsed:.2f}s (limit: {timeout}s) for dataset with {len(data)} samples"
        )
        return 0.0
    except Exception as e:
        # Other exceptions - treat as failed
        elapsed = time.time() - start_time
        logger.debug(f"Evaluation error after {elapsed:.2f}s: {e}")
        return 0.0


# -----------------------------
# AST mutation engine
# -----------------------------


@dataclass
class MutationSite:
    operator_name: str
    operator_cls: Any
    node: Any
    indices: List[int]


def _iter_tree(node):
    yield node
    if hasattr(node, "children") and node.children is not None:
        for child in node.children:
            # parso uses None for some whitespace slots; skip them
            if child is None:
                continue
            yield from _iter_tree(child)


def load_custom_operators():
    provider = Provider()
    operators = []
    for operator_name in Provider():
        operator_cls = provider[operator_name]
        operators.append((operator_name, operator_cls))
    return operators


def load_default_operators():
    from cosmic_ray.operators.provider import OperatorProvider

    provider = OperatorProvider()
    operators = []
    for operator_name in provider:
        if not any(
            operator_name.startswith(prefix)
            for prefix in {
                "ReplaceBinaryOperator",
                "ReplaceComparisonOperator",
                "NumberReplacer",
                "ReplaceUnaryOperator",
                "ZeroIterationForLoop",
                "AddNot",
            }
        ):
            continue
        operator_cls = provider[operator_name]
        operators.append((operator_name, operator_cls))
    return operators


def discover_mutation_sites(tree) -> List[MutationSite]:
    sites: List[MutationSite] = []
    operators = load_custom_operators()
    operators.extend(load_default_operators())
    for operator_name, operator_cls in operators:
        for node in _iter_tree(tree):
            try:
                operator_instance = operator_cls()
                positions = list(operator_instance.mutation_positions(node))
            except Exception:
                # Some operators may expect specific node interfaces; ignore failures
                continue
            if not positions:
                continue
            # Indices are 0..(len(positions)-1)
            sites.append(
                MutationSite(
                    operator_name=operator_name,
                    operator_cls=operator_cls,
                    node=node,
                    indices=list(range(len(positions))),
                )
            )
    return sites


def apply_mutation_once(source_code: str, rng: random.Random) -> Optional[str]:
    tree = parso.parse(source_code)
    sites = discover_mutation_sites(tree)
    if not sites:
        return None

    site = rng.choice(sites)
    index = rng.choice(site.indices)
    operator = site.operator_cls()

    # Apply the mutation. Some operators mutate in-place, others return a new node
    mutated_node = operator.mutate(site.node, index)
    if mutated_node is not site.node:
        # Replace in parent if possible
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

    return tree.get_code()


# -----------------------------
# GP core
# -----------------------------


@dataclass
class Candidate:
    source: str
    train_acc: float
    test_acc: float = None

    def key(self) -> Tuple[float, float]:
        return (self.train_acc, self.test_acc)


def hash_source(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()


def evaluate_source(
    source_code: str,
    program_name: str,
    max_length: int,
    accepted_inputs: Sequence[Any],
    train_data: List[Tuple[List[Any], List[Any]]],
    timeout: float = 30.0,
) -> Tuple[float, float]:
    logger.debug(f"Building program from source (timeout: {timeout}s)")
    build_start = time.time()
    try:
        program = build_program_from_source(
            source_code, program_name, max_length, accepted_inputs, timeout=timeout
        )
        build_elapsed = time.time() - build_start
        logger.debug(f"Program built successfully in {build_elapsed:.2f}s")
    except Exception as e:
        build_elapsed = time.time() - build_start
        logger.debug(f"Program building failed after {build_elapsed:.2f}s: {e}")
        raise
    
    logger.debug(f"Evaluating on train data ({len(train_data)} samples)")
    train_acc = evaluate_program_on_split(program, train_data, timeout=timeout)
    return train_acc


def evaluate_source_test(
    source_code: str,
    program_name: str,
    max_length: int,
    accepted_inputs: Sequence[Any],
    test_data: List[Tuple[List[Any], List[Any]]],
    timeout: float = 30.0,
) -> float:
    logger.debug(f"Building program from source for test (timeout: {timeout}s)")
    build_start = time.time()
    try:
        program = build_program_from_source(
            source_code, program_name, max_length, accepted_inputs, timeout=timeout
        )
        build_elapsed = time.time() - build_start
        logger.debug(f"Program built successfully in {build_elapsed:.2f}s")
    except Exception as e:
        build_elapsed = time.time() - build_start
        logger.debug(f"Program building failed after {build_elapsed:.2f}s: {e}")
        raise
    
    logger.debug(f"Evaluating on test data ({len(test_data)} samples)")
    return evaluate_program_on_split(program, test_data, timeout=timeout)


def run_gp(
    base_source: str,
    program_name_raw: str,
    data_dir: Path,
    accepted_inputs: Sequence[Any],
    budget: int = 500,
    population_size: int = 16,
    tournament_k: int = 3,
    seed: int = 42,
    log_every: int = 25,
    eval_timeout: float = 30.0,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    program_name = canonicalize_for_dataset(program_name_raw)

    # Load data and compute max length once
    train_data = load_dataset(data_dir, program_name, "train")
    val_data = load_dataset(data_dir, program_name, "val")
    # Fitness uses train+val merged
    fitness_data = train_data + val_data
    test_data = load_dataset(data_dir, program_name, "test")
    max_length = compute_max_length(data_dir, program_name)

    # Caches and bookkeeping
    seen: set[str] = set()
    cache: Dict[str, Tuple[float, float]] = {}
    num_programs = 0
    num_duplicates = 0
    num_compile_failures = 0

    def get_or_eval(src: str) -> Optional[Tuple[float, float]]:
        nonlocal num_programs, num_compile_failures, num_duplicates
        h = hash_source(src)
        if h in cache:
            num_duplicates += 1
            return cache[h]
        # Try to compile and evaluate; treat compile failures as invalid (not counted)
        eval_start = time.time()
        logger.debug(f"Evaluating candidate {h[:8]}... (program #{num_programs})")
        try:
            train_acc = evaluate_source(
                src,
                program_name,
                max_length,
                accepted_inputs,
                fitness_data,
                timeout=eval_timeout,
            )
            test_acc = evaluate_source_test(
                src,
                program_name,
                max_length,
                accepted_inputs,
                test_data,
                timeout=eval_timeout,
            )
            eval_elapsed = time.time() - eval_start
            logger.debug(f"Candidate {h[:8]}... evaluated in {eval_elapsed:.2f}s: train={train_acc:.4f}, test={test_acc:.4f}")
        except Exception as e:
            num_compile_failures += 1
            eval_elapsed = time.time() - eval_start
            logger.debug(f"Compile/eval failed for candidate {h[:8]}... after {eval_elapsed:.2f}s: {e}")
            return None
        cache[h] = (train_acc, test_acc)
        if log_every and num_programs % max(1, log_every) == 0:
            logger.info(
                f"Progress: programs={num_programs}/{budget} | dup={num_duplicates} | fails={num_compile_failures} | best_train={best_by_train.train_acc if 'best_by_train' in locals() else 'N/A'}"
            )
        return train_acc, test_acc

    # Initialize population with base + mutated variants
    population: List[Candidate] = []
    logger.info(
        f"Starting GP search for program={program_name} with budget={budget}, pop={population_size}, k={tournament_k}, seed={seed}"
    )
    # Track each program evaluation (like exhaustive search)
    program_history: List[Dict[str, Any]] = []

    # Track best seen so far
    best_train = -1.0
    best_test = -1.0

    # Count base program towards budget
    num_programs += 1
    base_eval = get_or_eval(base_source)
    if base_eval is not None:
        best_train = base_eval[0]
        best_test = base_eval[1]
        population.append(Candidate(base_source, base_eval[0], base_eval[1]))
        seen.add(hash_source(base_source))
        logger.info(
            f"Seeded base candidate: train={base_eval[0]:.4f} test={base_eval[1]:.4f}"
        )
        status = "ok"
        train_acc = base_eval[0]
        test_acc = base_eval[1]
    else:
        status = "compile_fail"
        train_acc = None
        test_acc = None

    generation = 0
    program_history.append(
        {
            "step": num_programs,
            "generation": generation,
            "status": status,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "best_train_acc": best_train,
            "best_test_acc": best_test,
        }
    )
    generation += 1

    # Fill remaining with random single-step mutations
    attempts = 0
    while (
        len(population) < population_size
        and attempts < population_size * 20
        and num_programs < budget
    ):
        attempts += 1
        mutated = apply_mutation_once(base_source, rng)
        if mutated is None:
            continue
        # Count this generated program regardless of later duplicate/compile status
        num_programs += 1
        h = hash_source(mutated)
        if h in seen:
            logger.debug(f"Duplicate candidate skipped: {h}")
            num_duplicates += 1
            status = "duplicate"
            eval_result = None
        else:
            eval_result = get_or_eval(mutated)
            if eval_result is None:
                status = "compile_fail"
            else:
                status = "ok"

        # Update best if we have a valid result
        if status == "ok" and eval_result is not None:
            train_acc = eval_result[0]
            test_acc = eval_result[1]
            if train_acc > best_train or (
                train_acc == best_train and test_acc > best_test
            ):
                best_train = train_acc
                best_test = test_acc
            seen.add(h)
            population.append(Candidate(mutated, train_acc, test_acc))
            logger.debug(
                f"Seeded mutated candidate {h}: train={train_acc:.4f} test={test_acc:.4f}"
            )
        else:
            train_acc = eval_result[0] if eval_result is not None else None
            test_acc = eval_result[1] if eval_result is not None else None

        program_history.append(
            {
                "step": num_programs,
                "generation": generation,
                "status": status,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "best_train_acc": best_train,
                "best_test_acc": best_test,
            }
        )

    if not population:
        # Could not seed a valid population
        return {
            "status": "failed_to_initialize",
            "num_programs": num_programs,
            "program_history": program_history,
        }

    # Track the best-by-train candidate globally
    best_by_train = max(population, key=lambda c: c.train_acc)
    best_train = best_by_train.train_acc
    best_test = best_by_train.test_acc
    logger.info(f"Initial best: train={best_by_train.train_acc:.4f}")

    generation += 1

    # Early stop if we already have perfect train
    if best_train >= 1.0:
        test_acc = evaluate_source_test(
            best_by_train.source,
            program_name,
            max_length,
            accepted_inputs,
            test_data,
            timeout=eval_timeout,
        )
        logger.info(
            f"Early stop: perfect train achieved after {num_programs} programs (dup={num_duplicates}, fails={num_compile_failures}). Test={test_acc:.4f}"
        )
        return {
            "status": "success_early",
            "num_programs": num_programs,
            "num_duplicates": num_duplicates,
            "num_compile_failures": num_compile_failures,
            "train_acc": best_train,
            "test_acc": test_acc,
            "best_source": best_by_train.source,
            "program_history": program_history,
        }

    # Main GP loop
    def tournament_select(pop: List[Candidate]) -> Candidate:
        contenders = rng.sample(pop, k=min(tournament_k, len(pop)))
        return max(contenders, key=lambda c: c.train_acc)

    while num_programs < budget:
        offspring: List[Candidate] = []
        generation += 1
        logger.info(
            f"Generation {generation} start: programs={num_programs}/{budget}, dup={num_duplicates}, fails={num_compile_failures}"
        )

        # Generate up to population_size offspring (μ=λ)
        gen_attempts = 0
        while (
            len(offspring) < population_size
            and num_programs < budget
            and gen_attempts < population_size * 50
        ):
            gen_attempts += 1
            parent = tournament_select(population)
            mutated = apply_mutation_once(parent.source, rng)
            if mutated is None:
                continue
            # Count this generated program regardless of later duplicate/compile status
            num_programs += 1
            h = hash_source(mutated)
            if h in seen:
                logger.debug(f"Duplicate candidate skipped: {h}")
                num_duplicates += 1
                status = "duplicate"
                train_acc = None
                test_acc = None
                # Get cached result if available
                if h in cache:
                    cached_train, cached_test = cache[h]
                    train_acc = cached_train
                    test_acc = cached_test
            else:
                eval_result = get_or_eval(mutated)
                if eval_result is None:
                    status = "compile_fail"
                    train_acc = None
                    test_acc = None
                else:
                    status = "ok"
                    train_acc = eval_result[0]
                    test_acc = eval_result[1]
                    seen.add(h)
                    cand = Candidate(mutated, train_acc, test_acc)
                    offspring.append(cand)
                    if train_acc > best_train or (
                        train_acc == best_train and test_acc > best_test
                    ):
                        best_train = train_acc
                        best_test = test_acc
                        best_by_train = cand
                        logger.info(
                            f"New best: train={best_train:.4f} test={best_test:.4f}"
                        )

            program_history.append(
                {
                    "step": num_programs,
                    "generation": generation,
                    "status": status,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "best_train_acc": best_train,
                    "best_test_acc": best_test,
                }
            )

            if best_train >= 1.0 or num_programs >= budget:
                break

        # Replacement: (μ+λ) keep top μ by train, tiebreak by test
        population = sorted(
            population + offspring, key=lambda c: c.train_acc, reverse=True
        )[:population_size]
        logger.info(
            f"Generation {generation} end: offspring={len(offspring)}, programs={num_programs}/{budget}, dup={num_duplicates}, fails={num_compile_failures}, best_train={best_train:.4f} | best_test={best_test:.4f}"
        )

        if best_train >= 1.0 or num_programs >= budget:
            break

    logger.info(
        f"Completed: programs={num_programs}/{budget}, dup={num_duplicates}, fails={num_compile_failures}, best_train={best_train:.4f}, test={best_test:.4f}"
    )

    return {
        "status": "complete",
        "num_programs": num_programs,
        "num_duplicates": num_duplicates,
        "num_compile_failures": num_compile_failures,
        "train_acc": best_train,
        "test_acc": best_test,
        "best_source": best_by_train.source,
        "program_history": program_history,
    }


def run_gp_for_bug(
    mutation_entry: Dict[str, Any],
    data_dir: Path,
    accepted_inputs: Sequence[Any],
    budget: int = 500,
    population_size: int = 60,
    tournament_k: int = 4,
    seed: int = 42,
    log_every: int = 25,
    eval_timeout: float = 30.0,
) -> Dict[str, Any]:
    program_name_raw = mutation_entry["program_name"]
    base_source = mutation_entry["program_source_after"]

    start_time = time.time()
    result = run_gp(
        base_source=base_source,
        program_name_raw=program_name_raw,
        data_dir=data_dir,
        accepted_inputs=accepted_inputs,
        budget=budget,
        population_size=population_size,
        tournament_k=tournament_k,
        seed=seed,
        log_every=log_every,
        eval_timeout=eval_timeout,
    )
    end_time = time.time()

    result.update(
        {
            "job_id": mutation_entry.get("job_id"),
            "program_name": mutation_entry.get("program_name"),
            "duration_s": end_time - start_time,
            "seed": seed,
        }
    )
    return result

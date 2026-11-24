"""Shared pieces of the symbolic repair baselines (GP and BFS).

Both baselines search the space of RASP *source* programs reachable from a buggy mutant
by applying the benchmark's own mutation operators (the `rasp_operators` plugin plus a
subset of cosmic-ray's default operators), and score each candidate by executing the
RASP program directly (no compilation to a Transformer) on the specification:

    fitness  = exact-sequence accuracy on train + val      (docs/FINDINGS.md F5)
    test_acc = exact-sequence accuracy on test (recorded for every candidate)

Every evaluation runs in a forked subprocess with a timeout; any failure (build error,
runtime error, timeout) scores 0.0 instead of raising.

Ported verbatim (behaviour-wise) from `experiments/gp_baseline/gp_core.py` and
`run_gp_exhaustive.py` at tag `tmlr-submission-1`.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

import parso

from rasping import programs
from rasping.data import load as load_split
from rasping.config import DATA_DIR

logger = logging.getLogger(__name__)

Example = Tuple[List[Any], List[Any]]

# --------------------------------------------------------------------------- data


def load_splits(program: str, data_dir: Path = DATA_DIR) -> tuple[list, list, list]:
    """The (train, val, test) specification of `program` (canonical or historical name)."""
    return (load_split(program, "train", data_dir), load_split(program, "val", data_dir),
            load_split(program, "test", data_dir))


def compute_max_length(program: str, data_dir: Path = DATA_DIR) -> int:
    """Longest input over train+val+test *including* BOS: the `max_len` v1 built
    candidates with (11 instead of the compiled models' 10; F11). No longer used by the
    searches, which use `config.MAX_LEN`."""
    train, val, test = load_splits(program, data_dir)
    return max(max(len(inp) for inp, _ in train + val + test), 1)


class CandidateError(Exception):
    """A candidate could not be evaluated: `kind` is "error" (build/eval failed) or "timeout"."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


# --------------------------------------------------------------------------- evaluation


def hash_source(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()


def build_program_from_source(source_code: str, program: str, max_length: int) -> Any:
    """Build (do not compile) the RASP SOp of `program` from candidate source code."""
    module_name = f"gp_candidate_{hashlib.sha1(source_code.encode('utf-8')).hexdigest()[:10]}"
    module = programs.module_from_source(source_code, module_name)
    return programs.get(program).build(module, max_length)


def evaluate_program_on_split(program: Any, data: List[Example]) -> float:
    """Sequence-level exact match of a RASP SOp, ignoring the BOS position.

    The SOp is run on `input[1:]` and compared with `target[1:]`; runtime errors count
    as incorrect.
    """
    if not data:
        return 0.0

    correct = 0
    total = 0
    start_time = time.time()
    for idx, (input_seq, target_seq) in enumerate(data):
        if len(data) > 10 and idx % max(1, len(data) // 10) == 0:
            logger.debug(f"Evaluating sample {idx}/{len(data)} "
                         f"(elapsed: {time.time() - start_time:.2f}s)")
        x = input_seq[1:]
        y = target_seq[1:]
        try:
            pred = program(x)
        except Exception:
            pred = None
        total += 1
        if pred is not None and list(pred) == list(y):
            correct += 1

    logger.debug(f"Completed evaluation of {total} samples in {time.time() - start_time:.2f}s")
    return correct / total if total > 0 else 0.0


def _evaluate_worker(result_queue, source_code: str, program: str, max_length: int,
                     data: List[Example]):
    """Build and evaluate a candidate inside the child process."""
    try:
        sop = build_program_from_source(source_code, program, max_length)
        result_queue.put(("success", evaluate_program_on_split(sop, data)))
    except Exception as e:
        result_queue.put(("error", e))


def evaluate_with_timeout(source_code: str, program: str, max_length: int,
                          data: List[Example], timeout: float = 30.0) -> float:
    """Accuracy of a candidate, evaluated in a separate process.

    Raises `CandidateError("error")` if the program cannot be built or evaluated and
    `CandidateError("timeout")` on timeout (v1 silently scored both as 0.0; F30).
    Runtime errors on individual inputs count as wrong predictions.
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_evaluate_worker,
        args=(result_queue, source_code, program, max_length, data),
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        logger.warning(f"Evaluation timeout after {timeout}s, terminating process")
        process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join()
        raise CandidateError("timeout", f"after {timeout}s")

    if not result_queue.empty():
        status, result = result_queue.get()
        if status == "success":
            return result
        raise CandidateError("error", repr(result))
    raise CandidateError("error", f"worker exited with code {process.exitcode}")


# Kept as the single entry point the searches call (v1 had two identical wrappers,
# `evaluate_source` for fitness and `evaluate_source_test` for test).
evaluate_source = evaluate_with_timeout


# --------------------------------------------------------------------------- mutation


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
            if child is None:  # parso uses None for some whitespace slots
                continue
            yield from _iter_tree(child)


#: cosmic-ray default operators used in addition to the RASP-specific ones.
DEFAULT_OPERATOR_PREFIXES = (
    "ReplaceBinaryOperator",
    "ReplaceComparisonOperator",
    "NumberReplacer",
    "ReplaceUnaryOperator",
    "ZeroIterationForLoop",
    "AddNot",
)


@lru_cache(maxsize=None)
def load_custom_operators() -> tuple:
    from rasp_operators.provider import Provider

    provider = Provider()
    return tuple((name, provider[name]) for name in Provider())


@lru_cache(maxsize=None)
def load_default_operators() -> tuple:
    from cosmic_ray.operators.provider import OperatorProvider

    provider = OperatorProvider()
    return tuple(
        (name, provider[name]) for name in provider
        if any(name.startswith(p) for p in DEFAULT_OPERATOR_PREFIXES)
    )


def discover_mutation_sites(tree) -> List[MutationSite]:
    """Every (operator, node) pair with at least one mutation position, in a fixed order:
    operators (custom first, then default) in provider order, nodes in tree pre-order."""
    sites: List[MutationSite] = []
    for operator_name, operator_cls in load_custom_operators() + load_default_operators():
        for node in _iter_tree(tree):
            try:
                positions = list(operator_cls().mutation_positions(node))
            except Exception:
                continue  # some operators expect specific node interfaces
            if not positions:
                continue
            sites.append(MutationSite(operator_name, operator_cls, node,
                                      list(range(len(positions)))))
    return sites


def _replace_in_parent(node, mutated_node) -> None:
    """Some operators mutate in place, others return a new node: splice the latter in."""
    if mutated_node is not node:
        parent = getattr(node, "parent", None)
        if parent is not None and hasattr(parent, "children") and parent.children is not None:
            for i, child in enumerate(parent.children):
                if child is node:
                    parent.children[i] = mutated_node
                    break


def apply_mutation_once(source_code: str, rng: random.Random) -> Optional[str]:
    """GP mutation: one random site, then one random position of it.

    May return the unchanged source (no-op mutations are not filtered); operator errors
    propagate.
    """
    tree = parso.parse(source_code)
    sites = discover_mutation_sites(tree)
    if not sites:
        return None

    site = rng.choice(sites)
    index = rng.choice(site.indices)
    operator = site.operator_cls()
    _replace_in_parent(site.node, operator.mutate(site.node, index))
    return tree.get_code()


def apply_mutation_by_index(source_code: str, site_index: int, position_index: int) -> Optional[str]:
    """BFS mutation: the `position_index`-th position of the `site_index`-th site.

    Returns None if the indices are invalid, anything fails, or the mutation is a no-op.
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
    _replace_in_parent(site.node, mutated_node)

    try:
        mutated_source = tree.get_code()
    except Exception:
        return None

    if mutated_source == source_code:
        return None
    return mutated_source


def all_single_step_neighbors(source_code: str) -> List[str]:
    """All single-step mutations of `source_code`, in site order (duplicates kept)."""
    try:
        tree = parso.parse(source_code)
    except Exception:
        return []

    neighbors: List[str] = []
    for site_idx, site in enumerate(discover_mutation_sites(tree)):
        for pos_idx in site.indices:
            mutated = apply_mutation_by_index(source_code, site_idx, pos_idx)
            if mutated:
                neighbors.append(mutated)
    return neighbors


def history_entry(step: int, phase_key: str, phase: int, status: str, train_acc, test_acc,
                  best_train: float, best_test: float, time_s: float | None = None) -> dict:
    """One `program_history` record (`phase_key` is "generation" for GP, "depth" for BFS).

    `best_train_acc` / `best_test_acc` are the fitness and test accuracy of the currently
    selected (best-fitness) program; `time_s` is seconds since the search started.
    """
    entry = {
        "step": step,
        phase_key: phase,
        "status": status,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "best_train_acc": best_train,
        "best_test_acc": best_test,
    }
    if time_s is not None:
        entry["time_s"] = round(time_s, 3)
    return entry

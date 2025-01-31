"""Evaluate generated patches and compute pass@1 metrics."""

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np

import re
import sys
import ast

module_paths = [
    str(Path(Path(__file__).parent.resolve(), "..", "..").resolve().absolute())
]
if module_paths not in sys.path:
    sys.path.extend(module_paths)

from src.model import Model
from src.functions import load_dataset
from src.jsonl import stream_jsonl, write_jsonl
from experiments.mutation.load_mutations import (
    create_module_from_source,
    getAcceptedNamesAndInput,
)


@dataclass
class EvaluationResult:
    passed: bool
    error: Optional[str] = None
    accuracy: Optional[float] = None


def evaluate_patch(patch: str, program_name: str, max_length: int) -> EvaluationResult:
    """Evaluate a single patch by creating a model and testing it.

    Args:
        patch: The patched source code
        program_name: Name of the program
        max_length: Maximum sequence length for the model
    """
    try:
        # First validate the patch is valid Python code
        ast.parse(patch)

        # Create module from patch
        module = create_module_from_source(patch, f"patch_{program_name}")

        # Get accepted inputs for this program
        program_name_key = program_name
        if program_name == "most_freq":
            program_name_key = "most-freq"
        elif program_name == "shuffle_dyck":
            program_name_key = "shuffle_dyck1"
        accepted_inputs = getAcceptedNamesAndInput()[program_name_key]
        inputs = {t for t in accepted_inputs}

        # Create and test the program
        try:
            # Create the patched program
            if program_name == "hist":
                program = module.make_hist()
            elif program_name == "sort":
                program = module.make_sort(min_key=min(inputs))
            elif program_name == "reverse":
                program = module.make_reverse()
            elif program_name == "most_freq":
                program = module.make_sort_freq(max_length)
            elif program_name == "shuffle_dyck":
                program = module.make_shuffle_dyck(["()"])
            elif program_name == "shuffle_dyck2":
                program = module.make_shuffle_dyck2()
            else:
                return EvaluationResult(
                    passed=False, error=f"Unknown program {program_name}"
                )

            # Create model from the patched program
            model = Model(program, inputs, max_length, program_name_key)

            # Load test data
            root_dir = Path(__file__).resolve().parents[2]
            test_data = load_dataset(root_dir / "data", program_name_key, "test")

            # Test the model on actual test data
            accuracy = model.evaluateModel(test_data, doPrint=False, outputArray=False)
            return EvaluationResult(passed=accuracy == 1.0, accuracy=accuracy)

        except Exception as e:
            return EvaluationResult(passed=False, error=str(e))

    except SyntaxError as e:
        return EvaluationResult(passed=False, error=f"Invalid Python syntax: {str(e)}")
    except Exception as e:
        return EvaluationResult(passed=False, error=str(e))


def extract_patch(response: dict) -> Optional[str]:
    # Get message from response object
    message = response["choices"][0]["message"]["content"]

    # Pattern to match code blocks with or without language specifier
    pattern = re.compile(r"```(\w*)\n([\s\S]*?)\n```")

    code_blocks = []
    for match in pattern.finditer(message):
        language = match.group(1)  # Capture the language specifier
        code = match.group(2)  # Capture the code block content
        code_blocks.append((language, code))

    return code_blocks[0][1] if code_blocks else None


def evaluate_single_patch_response(args) -> Dict[str, Any]:
    """Helper function to evaluate a single patch response.

    Args:
        args: Tuple of (response, program_name, max_length, patch_index)
    """
    response, program_name, max_length, patch_index = args
    patch = extract_patch(response)
    result = evaluate_patch(patch, program_name, max_length)
    return {
        "patch_index": patch_index,
        "passed": result.passed,
        "error": result.error,
        "accuracy": result.accuracy,
    }


def evaluate_patches_for_mutation(
    patches_file: Path,
    max_length: int = 10,
) -> Dict[str, Any]:
    """Evaluate all patches for a single mutation.

    Args:
        patches_file: Path to the JSONL file containing patches
        max_length: Maximum sequence length for testing
    """
    # Read first (and only) record from JSONL file
    data = next(stream_jsonl(str(patches_file)))

    eval_args = [
        (response, data["program_name"], max_length, i)
        for i, response in enumerate(data["responses"])
    ]

    results = [None] * len(eval_args)  # Pre-allocate results list
    n_workers = min(len(eval_args), multiprocessing.cpu_count())

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(evaluate_single_patch_response, arg): i
            for i, arg in enumerate(eval_args)
        }

        # Process results as they complete using as_completed
        with tqdm(
            total=len(eval_args), desc=f"Evaluating patches for {data['program_name']}"
        ) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    results[idx] = {
                        "patch_index": idx,
                        "passed": False,
                        "error": f"Process error: {str(e)}",
                        "accuracy": None,
                    }
                pbar.update(1)

    return {
        "mutation_id": data["mutation_id"],
        "program_name": data["program_name"],
        "job_id": data["job_id"],
        "patch_results": results,
    }


def pass_at_k(n: int, c: int, k: int):
    """Compute pass@k metric.

    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k
    """
    if n - c < k:
        return 1.0
    else:
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def generate_program_summary(
    results_dir: Path, program_name: str, output_file: Path
) -> None:
    """Generate a summary of all mutation results for a specific program."""
    program_results = []
    for result_file in results_dir.glob(f"{program_name}_*.jsonl"):
        mutation_result = next(stream_jsonl(str(result_file)))
        program_results.append(mutation_result)

    total_mutations = len(program_results)
    fixed_mutations = sum(
        1
        for result in program_results
        if any(patch["passed"] for patch in result["patch_results"])
    )

    # Count total patches and correct patches across all mutations
    total_patches = sum(len(result["patch_results"]) for result in program_results)
    correct_patches = sum(
        sum(1 for patch in result["patch_results"] if patch["passed"])
        for result in program_results
    )

    # Calculate pass@1 using total numbers
    pass_at_1 = pass_at_k(total_patches, correct_patches, 1)

    summary = {
        "program_name": program_name,
        "total_mutations": total_mutations,
        "fixed_mutations": fixed_mutations,
        "total_patches": total_patches,
        "correct_patches": correct_patches,
        "pass_at_1": pass_at_1,
    }
    write_jsonl(str(output_file), [summary])


def generate_overall_summary(summaries_dir: Path, output_file: Path) -> None:
    """Generate an overall summary of all program results."""
    program_summaries = []
    for summary_file in summaries_dir.glob("*_summary.jsonl"):
        program_summary = next(stream_jsonl(str(summary_file)))
        program_summaries.append(program_summary)

    total_mutations = sum(summary["total_mutations"] for summary in program_summaries)
    total_fixed = sum(summary["fixed_mutations"] for summary in program_summaries)

    # Sum all patches and correct patches across all programs
    total_patches = sum(summary["total_patches"] for summary in program_summaries)
    total_correct = sum(summary["correct_patches"] for summary in program_summaries)

    # Calculate overall pass@1
    pass_at_1 = pass_at_k(total_patches, total_correct, 1)

    overall_summary = {
        "total_programs": len(program_summaries),
        "total_mutations": total_mutations,
        "total_fixed_mutations": total_fixed,
        "total_patches": total_patches,
        "correct_patches": total_correct,
        "pass_at_1": pass_at_1,
    }
    write_jsonl(str(output_file), [overall_summary])


def evaluate_all_patches(
    patches_dir: str, output_dir: str, max_length: int = 10
) -> None:
    """Evaluate all patches and compute aggregate metrics.

    Args:
        patches_dir: Directory containing patch JSON files
        output_dir: Directory to save evaluation results
        max_length: Maximum sequence length for testing
    """
    patches_path = Path(patches_dir)
    output_path = Path(output_dir)
    results_dir = output_path / "mutation_results"
    summaries_dir = output_path / "program_summaries"

    # Create necessary directories
    results_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    patch_files = list(patches_path.glob("*.jsonl"))
    program_results = {}

    # Process patch files sequentially but parallelize patch evaluation within each file
    for patch_file in tqdm(patch_files, desc="Processing patch files"):
        result = evaluate_patches_for_mutation(patch_file, max_length)
        program_name = result["program_name"]

        # Save individual mutation result
        output_file = results_dir / f"{program_name}_{result['job_id']}.jsonl"
        write_jsonl(str(output_file), [result])

        # Collect results by program
        if program_name not in program_results:
            program_results[program_name] = []
        program_results[program_name].append(result)

    # Generate program summaries
    for program_name in program_results:
        summary_file = summaries_dir / f"{program_name}_summary.jsonl"
        generate_program_summary(results_dir, program_name, summary_file)

    # Generate overall summary
    overall_summary_file = output_path / "overall_summary.jsonl"
    generate_overall_summary(summaries_dir, overall_summary_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patches-dir",
        default=str(Path(__file__).parent / "results"),
        help="Directory containing patch JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "evaluation_results"),
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--max-length", type=int, default=10, help="Maximum sequence length for testing"
    )
    args = parser.parse_args()

    evaluate_all_patches(
        patches_dir=args.patches_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
    )

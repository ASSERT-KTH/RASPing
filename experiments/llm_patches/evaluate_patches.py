"""Evaluate generated patches and compute pass@1 metrics."""

import re
import sys
from pathlib import Path
import json
import ast
from typing import Dict, Any, Optional
from dataclasses import dataclass

module_paths = [
    str(Path(Path(__file__).parent.resolve(), "..", "..").resolve().absolute())
]
if module_paths not in sys.path:
    sys.path.extend(module_paths)

from src.model import Model
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

            model = Model(program, inputs, max_length, program_name)
            accuracy = model.evaluateValidationData()
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


def evaluate_patches_for_mutation(
    patches_file: Path,
    max_length: int = 10,
) -> Dict[str, Any]:
    """Evaluate all patches for a single mutation.

    Args:
        patches_file: Path to the JSON file containing patches
        max_length: Maximum sequence length for testing
    """
    with open(patches_file) as f:
        data = json.load(f)

    results = []
    for i, response in enumerate(data["responses"]):
        patch = extract_patch(response)
        result = evaluate_patch(patch, data["program_name"], max_length)
        results.append(
            {
                "patch_index": i,
                "passed": result.passed,
                "error": result.error,
                "accuracy": result.accuracy,
            }
        )

    # Calculate pass@1
    pass_at_1 = any(r["passed"] for r in results[:1])

    return {
        "mutation_id": data["mutation_id"],
        "program_name": data["program_name"],
        "job_id": data["job_id"],
        "pass_at_1": pass_at_1,
        "patch_results": results,
    }


def evaluate_all_patches(
    patches_dir: str, output_file: str, max_length: int = 10
) -> None:
    """Evaluate all patches and compute aggregate metrics.

    Args:
        patches_dir: Directory containing patch JSON files
        output_file: Path to save evaluation results
        max_length: Maximum sequence length for testing
    """
    patches_path = Path(patches_dir)
    results = []

    for patch_file in patches_path.glob("*.json"):
        result = evaluate_patches_for_mutation(patch_file, max_length)
        results.append(result)

    # Calculate aggregate metrics
    total_mutations = len(results)
    if total_mutations > 0:
        pass_at_1 = sum(1 for r in results if r["pass_at_1"]) / total_mutations
    else:
        pass_at_1 = 0.0

    # Group by program
    by_program = {}
    for r in results:
        prog = r["program_name"]
        if prog not in by_program:
            by_program[prog] = {"total": 0, "pass_at_1": 0}
        by_program[prog]["total"] += 1
        if r["pass_at_1"]:
            by_program[prog]["pass_at_1"] += 1

    for prog in by_program:
        by_program[prog]["pass_at_1_rate"] = (
            by_program[prog]["pass_at_1"] / by_program[prog]["total"]
            if by_program[prog]["total"] > 0
            else 0.0
        )

    # Save results
    with open(output_file, "w") as f:
        json.dump(
            {
                "overall_pass_at_1": pass_at_1,
                "by_program": by_program,
                "individual_results": results,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patches-dir",
        default=str(Path(__file__).parent / "results"),
        help="Directory containing patch JSON files",
    )
    parser.add_argument(
        "--output-file",
        default=str(Path(__file__).parent / "evaluation_results.json"),
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--max-length", type=int, default=10, help="Maximum sequence length for testing"
    )
    args = parser.parse_args()

    evaluate_all_patches(
        patches_dir=args.patches_dir,
        output_file=args.output_file,
        max_length=args.max_length,
    )

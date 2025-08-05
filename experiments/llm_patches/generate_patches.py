"""Generate prompts and collect patches from LLM for each mutation."""

import sys
from pathlib import Path
from typing import Dict, Any
import openai
import backoff
import random
from tqdm import tqdm
import click
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

module_paths = [
    str(Path(Path(__file__).parent.resolve(), "..", "..").resolve().absolute())
]
if module_paths not in sys.path:
    sys.path.extend(module_paths)

from experiments.mutation.load_mutations import load_mutation, create_model_from_mutation, getAcceptedNamesAndInput
from experiments.llm_patches.prompts import build_prompt
from src.jsonl import write_jsonl
from src.functions import load_dataset


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    raise_on_giveup=True,
)
def generate_completion(client, **kwargs):
    """Generate completion with backoff retry logic."""
    return client.chat.completions.create(**kwargs)



def _compute_example_behaviors(mutation: Dict[str, Any], max_length: int = 10):
    """Find an input example where the buggy program fails and return expected and current behavior strings."""
    try:
        # Build buggy and correct models
        buggy_model = create_model_from_mutation(mutation, max_length)

        # Create a copy of the mutation dict with the *correct* program source
        correct_mutation = mutation.copy()
        if "program_source_before" not in mutation:
            # If we don't have the original source, we cannot compute the example
            return None, None, None
        correct_mutation["program_source_after"] = mutation["program_source_before"]
        correct_model = create_model_from_mutation(correct_mutation, max_length)

        # Map program name to dataset key used in load_dataset
        program_name_key = mutation["program_name"]
        if program_name_key == "most_freq":
            program_name_key = "most-freq"
        elif program_name_key == "shuffle_dyck":
            program_name_key = "shuffle_dyck1"

        # Load validation dataset
        data_path = Path(__file__).parent.parent.parent / "data"
        try:
            val_dataset = load_dataset(str(data_path), program_name_key, split_name="val")
        except Exception:
            val_dataset = []

        # Helper to stringify sequences
        def _clean(seq):
            return [str(tok) for tok in seq if str(tok).upper() != "BOS"]

        # 1. Look in validation dataset for a failing example
        for input_seq, _ in val_dataset:
            expected_output = _clean(correct_model.apply(input_seq))
            current_output = _clean(buggy_model.apply(input_seq))
            if expected_output != current_output:
                input_tokens = _clean(input_seq)
                return (
                    " ".join(input_tokens),
                    " ".join(expected_output),
                    " ".join(current_output),
                )

        # 2. If none found, sample random sequences
        accepted_inputs = getAcceptedNamesAndInput().get(program_name_key, [])
        if not accepted_inputs:
            return None, None, None
        for _ in range(100):
            length = random.randint(1, max_length)
            input_seq = random.choices(accepted_inputs, k=length)
            expected_output = _clean(correct_model.apply(input_seq))
            current_output = _clean(buggy_model.apply(input_seq))
            if expected_output != current_output:
                return (
                    " ".join([str(tok) for tok in input_seq]),
                    " ".join(expected_output),
                    " ".join(current_output),
                )
    except Exception as e:
        # If anything fails, just return None so we don't break the pipeline
        print(f"Warning: Failed to compute example behaviors for mutation {mutation.get('job_id')}: {e}")
    return None, None, None


def generate_patches_for_mutation(
    api_key: str,
    mutation: Dict[str, Any],
    n_patches: int = 1,
    model_name: str = "gpt-4.1-2025-04-14",
    temperature: float = 0.2,
    prompt_type: str = "small",
) -> list[Dict[str, Any]]:
    """Generate patches for a single mutation using the OpenAI API."""

    # Compute example behaviors to include in the prompt
    example_input, expected_behavior, current_behavior = _compute_example_behaviors(mutation)
    if expected_behavior and current_behavior:
        mutation["expected_behavior"] = f"Input: {example_input}\nOutput: {expected_behavior}"
        mutation["current_behavior"] = f"Input: {example_input}\nOutput: {current_behavior}"
    else:
        raise ValueError("No example behaviors found")

    prompt = build_prompt(
        mutation=mutation,
        prompt_type=prompt_type,
    )
    
    print(prompt)

    client = openai.OpenAI(api_key=api_key)
    responses = []
    completion = generate_completion(
        client,
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        n=n_patches,
    )
    responses.append(completion.to_dict())

    return responses


def process_single_mutation(args):
    """Process a single mutation with its arguments."""
    (
        idx,
        mutation,
        api_key,
        n_patches,
        output_path,
        model_name,
        temperature,
        prompt_type,
    ) = args
    result = generate_patches_for_mutation(
        api_key,
        mutation,
        n_patches,
        model_name=model_name,
        temperature=temperature,
        prompt_type=prompt_type,
    )

    # Save results immediately after generation as JSONL
    output_file = output_path / f"{mutation['program_name']}_{mutation['job_id']}.jsonl"
    write_jsonl(
        str(output_file),
        [
            {
                "mutation_id": idx,
                "program_name": mutation["program_name"],
                "job_id": mutation["job_id"],
                "responses": result,
                "prompt": build_prompt(mutation, prompt_type),
                "model_name": model_name,
                "temperature": temperature,
            }
        ],
    )

    return idx, result


def generate_all_patches(
    mutation_path: str,
    output_dir: str,
    api_key: str,
    n_patches: int = 1,
    program_name: str = None,
    job_id: str = None,
    model_name: str = "gpt-4o-mini-2024-07-18",
    temperature: float = 0.2,
    prompt_type: str = "small",
) -> None:
    """Generate patches for all mutations matching the filters."""
    mutations = load_mutation(mutation_path, program_name, job_id)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for parallel processing
    args_list = [
        (
            idx,
            mutation,
            api_key,
            n_patches,
            output_path,
            model_name,
            temperature,
            prompt_type,
        )
        for idx, mutation in mutations.items()
    ][:1]

    # Use ThreadPoolExecutor for parallel processing
    # Number of workers is min of CPU count and number of mutations
    max_workers = min(len(args_list), 2 * os.cpu_count() or 1)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show progress
        futures = list(
            tqdm(
                executor.map(process_single_mutation, args_list),
                total=len(args_list),
                desc="Generating patches",
            )
        )
        results.extend(futures)


@click.command()
@click.option(
    "--n-patches",
    default=1,
    help="Number of patches to generate per mutation",
    type=int,
)
@click.option(
    "--program-name", default=None, help="Only generate patches for this program"
)
@click.option("--job-id", default=None, help="Only generate patches for this job ID")
@click.option(
    "--mutation-path",
    default=lambda: str(
        Path(__file__).parent.parent / "mutation/raspbugs.json"
    ),
    help="Path to mutations JSON file",
)
@click.option(
    "--output-dir",
    default=lambda: str(Path(__file__).parent / "results"),
    help="Directory to save generated patches",
)
@click.option(
    "--model-name",
    default="gpt-4o-mini-2024-07-18",
    help="Model name to use for generation",
)
@click.option(
    "--temperature",
    default=0.2,
    help="Temperature for generation",
    type=float,
)
@click.option(
    "--prompt-type",
    default="small",
    help="Prompt type to use for generation",
    type=str,
)
def main(
    n_patches: int,
    program_name: str,
    job_id: str,
    mutation_path: str,
    output_dir: str,
    model_name: str,
    temperature: float,
    prompt_type: str,
) -> None:
    """Generate patches for mutations using OpenAI API."""
    # Use API key from environment if not provided via CLI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise click.UsageError(
            "API key must be provided via OPENAI_API_KEY environment variable"
        )

    generate_all_patches(
        mutation_path=mutation_path,
        output_dir=output_dir,
        api_key=api_key,
        n_patches=n_patches,
        program_name=program_name,
        job_id=job_id,
        model_name=model_name,
        temperature=temperature,
        prompt_type=prompt_type,
    )


if __name__ == "__main__":
    main()

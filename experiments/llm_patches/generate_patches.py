"""Generate prompts and collect patches from LLM for each mutation."""

import sys
from pathlib import Path
import json
from typing import Dict, Any
import requests
import backoff
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

from experiments.mutation.load_mutations import load_mutation
from experiments.llm_patches.prompts import build_prompt
from src.jsonl import write_jsonl


class OpenRouterClient:
    def __init__(self, api_key: str):
        self.openrouter_api_key = api_key

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError, Exception),
        max_tries=5,
        raise_on_giveup=False,
    )
    def _completions_with_backoff(self, **kwargs):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                # For including your app on openrouter.ai rankings.
                "HTTP-Referer": f"https://repairbench.github.io/",
                # Shows in rankings on openrouter.ai.
                "X-Title": f"RepairBench",
            },
            data=json.dumps(kwargs),
        )

        response = response.json()

        if "error" in response:
            raise Exception(response["error"])

        return response


def generate_patches_for_mutation(
    api_key: str,
    mutation: Dict[str, Any],
    n_patches: int = 1,
    model_name: str = "deepseek/deepseek-r1",
    provider: str = "DeepSeek",
    temperature: float = 0.2,
) -> list[Dict[str, Any]]:
    """Generate patches for a single mutation using the OpenRouter API."""
    prompt = build_prompt(
        buggy_program=mutation["program_source_after"],
    )

    client = OpenRouterClient(api_key)
    responses = []
    for _ in range(n_patches):
        try:
            response = client._completions_with_backoff(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                provider={
                    "require_parameters": False,
                    "allow_fallbacks": False,
                    "order": [provider],
                },
                include_reasoning=True,
            )
            responses.append(response)
        except Exception as e:
            print(f"Failed to generate patch: {str(e)}")
            responses.append({"error": str(e)})

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
        provider,
        temperature,
    ) = args
    result = generate_patches_for_mutation(
        api_key,
        mutation,
        n_patches,
        model_name=model_name,
        provider=provider,
        temperature=temperature,
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
                "prompt": build_prompt(mutation["program_source_after"]),
                "model_name": model_name,
                "provider": provider,
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
    model_name: str = "deepseek/deepseek-r1",
    provider: str = "DeepSeek",
    temperature: float = 0.2,
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
            provider,
            temperature,
        )
        for idx, mutation in mutations.items()
    ]

    # Use ThreadPoolExecutor for parallel processing
    # Number of workers is min of CPU count and number of mutations
    max_workers = min(len(args_list), os.cpu_count() or 1)

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
        Path(__file__).parent.parent / "mutation/results/aggregated_mutations.json"
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
    default="deepseek/deepseek-r1",
    help="Model name to use for generation",
)
@click.option(
    "--provider",
    default="DeepSeek",
    help="Provider to use for generation",
)
@click.option(
    "--temperature",
    default=0.2,
    help="Temperature for generation",
    type=float,
)
def main(
    n_patches: int,
    program_name: str,
    job_id: str,
    mutation_path: str,
    output_dir: str,
    model_name: str,
    provider: str,
    temperature: float,
) -> None:
    """Generate patches for mutations using OpenRouter API."""
    # Use API key from environment if not provided via CLI
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise click.UsageError(
            "API key must be provided via OPENROUTER_API_KEY environment variable"
        )

    generate_all_patches(
        mutation_path=mutation_path,
        output_dir=output_dir,
        api_key=api_key,
        n_patches=n_patches,
        program_name=program_name,
        job_id=job_id,
        model_name=model_name,
        provider=provider,
        temperature=temperature,
    )


if __name__ == "__main__":
    main()

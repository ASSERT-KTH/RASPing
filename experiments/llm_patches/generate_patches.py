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

# Load environment variables
load_dotenv()

module_paths = [
    str(Path(Path(__file__).parent.resolve(), "..", "..").resolve().absolute())
]
if module_paths not in sys.path:
    sys.path.extend(module_paths)

from experiments.mutation.load_mutations import load_mutation
from experiments.llm_patches.prompts import build_prompt


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
                model="deepseek/deepseek-r1",
                messages=[{"role": "user", "content": prompt}],
                provider={
                    "require_parameters": True,
                    "allow_fallbacks": False,
                    "order": ["DeepSeek"],
                },
                include_reasoning=True,
            )
            responses.append(response)
        except Exception as e:
            print(f"Failed to generate patch: {str(e)}")
            responses.append({"error": str(e)})

    return responses


def generate_all_patches(
    mutation_path: str,
    output_dir: str,
    api_key: str,
    n_patches: int = 1,
    program_name: str = None,
    job_id: str = None,
) -> None:
    """Generate patches for all mutations matching the filters."""
    mutations = load_mutation(mutation_path, program_name, job_id)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, mutation in tqdm(mutations.items()):
        result = generate_patches_for_mutation(api_key, mutation, n_patches)
        results.append(result)

        # Save results immediately after generation
        output_file = (
            output_path / f"{mutation['program_name']}_{mutation['job_id']}.json"
        )
        with open(output_file, "w") as f:
            json.dump(
                {
                    "mutation_id": idx,
                    "program_name": mutation["program_name"],
                    "job_id": mutation["job_id"],
                    "responses": result,
                    "prompt": build_prompt(mutation["program_source_after"]),
                },
                f,
                indent=2,
            )


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
def main(
    n_patches: int, program_name: str, job_id: str, mutation_path: str, output_dir: str
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
    )


if __name__ == "__main__":
    main()

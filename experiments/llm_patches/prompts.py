from pathlib import Path


def build_prompt(mutation: dict, prompt_type: str) -> str:
    prompt_path = Path(__file__).parent / f"prompt_{prompt_type}.txt"
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

        # Replace placeholders with actual values
        prompt = prompt_template.replace("{{PROGRAM_HERE}}", mutation["program_source_after"])
        prompt = prompt.replace("{{EXPECTED_BEHAVIOR}}", mutation["expected_behavior"])
        prompt = prompt.replace("{{CURRENT_BEHAVIOR}}", mutation["current_behavior"])

        return prompt

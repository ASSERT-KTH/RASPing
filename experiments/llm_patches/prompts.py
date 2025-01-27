from pathlib import Path


def build_prompt(buggy_program: str) -> str:
    prompt_path = Path(__file__).parent / "prompt.txt"
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
        return prompt_template.replace("<PROGRAM_HERE>", buggy_program)

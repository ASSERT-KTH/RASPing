import os
import shutil
import toml
import subprocess
from pathlib import Path
from multiprocessing import Pool
from typing import Optional


def create_config(source_file: str) -> str:
    """Create a cosmic-ray config file for a given source file."""

    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(source_file))[0]

    config = {
        "cosmic-ray": {
            "module-path": f"source/{os.path.basename(source_file)}",
            "timeout": 60.0,
            "excluded-modules": [],
            "test-command": f"pytest -s tests/test_{base_name}.py",
            "distributor": {"name": "local"},
        }
    }

    # Create results directory if it doesn't exist
    results_dir = Path("results") / base_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create config filename in the results directory
    config_file = results_dir / f"tracr_{base_name}.toml"

    # Write config to file
    with open(config_file, "w") as f:
        toml.dump(config, f)

    return config_file


def process_file(source_file: Path) -> Optional[str]:
    """Process a single source file with cosmic-ray."""
    try:
        print(f"Processing {source_file}...")

        # Cleanup results directory
        results_dir = Path("results") / source_file.stem
        if results_dir.exists():
            shutil.rmtree(results_dir)

        # Create config file
        config_file = create_config(str(source_file))

        # Initialize the session
        sqlite_file = results_dir / f"tracr_{source_file.stem}.sqlite"
        subprocess.run(
            ["cosmic-ray", "init", str(config_file), str(sqlite_file)], check=True
        )

        # Run the mutations
        subprocess.run(
            ["cosmic-ray", "exec", str(config_file), str(sqlite_file)], check=True
        )

        # Dump to JSON
        jsonl_file = results_dir / f"tracr_{source_file.stem}.jsonl"
        with open(jsonl_file, "w") as f:
            subprocess.run(
                ["cosmic-ray", "dump", str(sqlite_file)], stdout=f, check=True
            )

        # Generate HTML report
        html_file = results_dir / f"report_{source_file.stem}.html"
        with open(html_file, "w") as f:
            subprocess.run(["cr-html", str(sqlite_file)], stdout=f, check=True)

        print(f"Completed mutation testing for {source_file}")
        return None

    except subprocess.CalledProcessError as e:
        return f"Error processing {source_file}: {e}"


def main():
    # Create main results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)

    # Get all Python files in source directory
    source_dir = Path("source")
    source_files = [f for f in source_dir.glob("*.py") if f.name != "__init__.py"]

    # Process files in parallel
    with Pool() as pool:
        errors = list(filter(None, pool.map(process_file, source_files)))

    # Print any errors that occurred
    for error in errors:
        print(error)


if __name__ == "__main__":
    main()

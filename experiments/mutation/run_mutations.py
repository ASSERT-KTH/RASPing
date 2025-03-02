import os
import shutil
import toml
import subprocess
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, Tuple, List
import argparse


def create_config(source_file: str, mutation_order: int = 1) -> str:
    """Create a cosmic-ray config file for a given source file.

    Args:
        source_file: Path to the source file to mutate
        mutation_order: Order of mutations to generate (default: 1)
    """

    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(source_file))[0]

    config = {
        "cosmic-ray": {
            "module-path": f"source/{os.path.basename(source_file)}",
            "timeout": 60.0,
            "excluded-modules": [],
            "test-command": f"pytest -s tests/test_{base_name}.py",
            "distributor": {"name": "local"},
            # Add higher-order mutation parameters
            "mutation-order": mutation_order,
            "specific-order": mutation_order,  # Only generate mutations of this exact order
            "mutation-limit": 50,  # Limit the number of mutations to avoid explosion
            "disable-overlapping-mutations": True,  # Prevent mutations from overlapping
        }
    }

    # Create results directory if it doesn't exist
    results_dir = Path("results") / base_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create config filename in the results directory
    config_file = results_dir / f"tracr_{base_name}_order={mutation_order}.toml"

    # Write config to file
    with open(config_file, "w") as f:
        toml.dump(config, f)

    return config_file


def process_file_with_order(args: Tuple[Path, int]) -> Optional[str]:
    """Process a single source file with cosmic-ray using specified mutation order.

    Args:
        args: Tuple containing (source_file, mutation_order)
    """
    source_file, mutation_order = args

    try:
        print(f"Processing {source_file} with mutation order {mutation_order}...")

        # Get the base filename without extension
        base_name = source_file.stem

        # Create results directory if it doesn't exist
        results_dir = Path("results") / base_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create config file for this mutation order
        config_file = create_config(str(source_file), mutation_order)

        # Create unique filenames for this mutation order
        sqlite_file = results_dir / f"tracr_{base_name}_order={mutation_order}.sqlite"
        jsonl_file = results_dir / f"tracr_{base_name}_order={mutation_order}.jsonl"
        html_file = results_dir / f"report_{base_name}_order={mutation_order}.html"

        # Initialize the session
        subprocess.run(
            ["cosmic-ray", "init", str(config_file), str(sqlite_file)], check=True
        )

        # Run the mutations
        subprocess.run(
            ["cosmic-ray", "exec", str(config_file), str(sqlite_file)], check=True
        )

        # Dump to JSON
        with open(jsonl_file, "w") as f:
            subprocess.run(
                ["cosmic-ray", "dump", str(sqlite_file)], stdout=f, check=True
            )

        # Generate HTML report
        with open(html_file, "w") as f:
            subprocess.run(["cr-html", str(sqlite_file)], stdout=f, check=True)

        print(
            f"Completed mutation testing for {source_file} with order {mutation_order}"
        )
        return None

    except subprocess.CalledProcessError as e:
        return f"Error processing {source_file} with order {mutation_order}: {e}"


def process_file(source_file: Path) -> Optional[str]:
    """Process a single source file with cosmic-ray."""
    # This function is kept for backward compatibility
    return process_file_with_order((source_file, 1))


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run mutation testing with different mutation orders"
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[1],
        help="Mutation orders to run (default: [1])",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean results directory before running"
    )
    args = parser.parse_args()

    # Create main results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)

    # Get all Python files in source directory
    source_dir = Path("source")
    source_files = [
        f
        for f in source_dir.glob("*.py")
        if f.name != "__init__.py" and "hist" in f.name
    ]

    # Clean results directories if requested
    if args.clean:
        for source_file in source_files:
            results_dir = Path("results") / source_file.stem
            if results_dir.exists():
                print(f"Cleaning {results_dir}...")
                shutil.rmtree(results_dir)

    # Create a list of (source_file, mutation_order) tuples for all combinations
    tasks: List[Tuple[Path, int]] = []
    for source_file in source_files:
        for order in args.orders:
            tasks.append((source_file, order))

    # Process files in parallel
    with Pool() as pool:
        errors = list(filter(None, pool.map(process_file_with_order, tasks)))

    # Print any errors that occurred
    for error in errors:
        print(error)


if __name__ == "__main__":
    main()

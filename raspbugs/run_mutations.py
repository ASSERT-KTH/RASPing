import os
import shutil
import toml
import subprocess
import time
from pathlib import Path
from multiprocessing import Pool, Manager, Value
from typing import Optional, Tuple, List, Dict
import argparse
from tqdm import tqdm


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
            "mutation-limit": 200,  # Limit the number of mutations to avoid explosion
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


def process_file_with_order(args: Tuple[Path, int, Dict]) -> Optional[str]:
    """Process a single source file with cosmic-ray using specified mutation order.

    Args:
        args: Tuple containing (source_file, mutation_order, shared_dict)
    """
    source_file, mutation_order, shared_dict = args
    process_id = os.getpid()
    program_name = source_file.stem

    try:
        # Instead of updating a shared progress bar, we'll print status updates
        print(f"Processing {program_name} (order={mutation_order}, PID={process_id})")
        
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
        init_result = subprocess.run(
            ["cosmic-ray", "init", str(config_file), str(sqlite_file)], 
            check=True
        )
        
        # Run the mutations
        exec_result = subprocess.run(
            ["cosmic-ray", "exec", str(config_file), str(sqlite_file)], 
            check=True
        )

        # Dump to JSON
        with open(jsonl_file, "w") as f:
            # Can't use capture_output with stdout specified
            dump_result = subprocess.run(
                ["cosmic-ray", "dump", str(sqlite_file)], 
                stdout=f, 
                check=True
            )

        # Generate HTML report
        with open(html_file, "w") as f:
            # Can't use capture_output with stdout specified
            html_result = subprocess.run(
                ["cr-html", str(sqlite_file)], 
                stdout=f, 
                check=True
            )

        # Increment completed tasks in the shared dictionary
        with shared_dict["lock"]:
            shared_dict["completed"] += 1
            print(f"Progress: {shared_dict['completed']}/{shared_dict['total']} tasks completed")

        return None

    except subprocess.CalledProcessError as e:
        error_msg = f"ERROR in {program_name} (order={mutation_order}, PID={process_id}): {e}\n"
        error_msg += f"Command: {e.cmd}\n"
        if hasattr(e, 'stdout') and e.stdout:
            error_msg += f"STDOUT: {e.stdout.decode('utf-8')}\n"
        if hasattr(e, 'stderr') and e.stderr:
            error_msg += f"STDERR: {e.stderr.decode('utf-8')}\n"
        print(error_msg)
        
        # Increment completed tasks in the shared dictionary
        with shared_dict["lock"]:
            shared_dict["completed"] += 1
            print(f"Progress: {shared_dict['completed']}/{shared_dict['total']} tasks completed")
            
        return error_msg
    except Exception as e:
        error_msg = f"EXCEPTION in {program_name} (order={mutation_order}, PID={process_id}): {e}"
        print(error_msg)
        
        # Increment completed tasks in the shared dictionary
        with shared_dict["lock"]:
            shared_dict["completed"] += 1
            print(f"Progress: {shared_dict['completed']}/{shared_dict['total']} tasks completed")
            
        return error_msg


def process_file(source_file: Path) -> Optional[str]:
    """Process a single source file with cosmic-ray."""
    # This function is kept for backward compatibility
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict["completed"] = 0
    shared_dict["total"] = 1
    shared_dict["lock"] = manager.Lock()
    return process_file_with_order((source_file, 1, shared_dict))


def process_program_sequentially(args: Tuple[Path, List[Tuple[Path, int]], Dict]) -> List[Optional[str]]:
    """Process all tasks for a single program sequentially.
    
    Args:
        args: Tuple containing (source_file, list_of_tasks, shared_dict)
    
    Returns:
        List of error messages, if any
    """
    source_file, tasks, shared_dict = args
    program_name = source_file.stem
    print(f"Starting sequential processing for program: {program_name} with {len(tasks)} tasks")
    
    errors = []
    for task in tasks:
        source_file, mutation_order = task
        error = process_file_with_order((source_file, mutation_order, shared_dict))
        if error:
            errors.append(error)
    return errors


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
    source_files = [f for f in source_dir.glob("*.py") if f.name != "__init__.py"]

    # Clean results directories if requested
    if args.clean:
        for source_file in source_files:
            results_dir = Path("results") / source_file.stem
            if results_dir.exists():
                print(f"Cleaning {results_dir}...")
                shutil.rmtree(results_dir)

    # Group tasks by program (source file)
    program_tasks: Dict[Path, List[Tuple[Path, int]]] = {}
    for source_file in source_files:
        program_tasks[source_file] = []
        for order in args.orders:
            program_tasks[source_file].append((source_file, order))

    # Calculate total number of tasks
    total_tasks = sum(len(tasks) for tasks in program_tasks.values())
    print(f"Total tasks to process: {total_tasks}")
    
    # Create shared dictionary for tracking progress across processes
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict["completed"] = 0
    shared_dict["total"] = total_tasks
    shared_dict["lock"] = manager.Lock()
    
    # Process each program's tasks in parallel (but tasks within a program sequentially)
    pool_input = [(file, tasks, shared_dict) for file, tasks in program_tasks.items()]
    
    # Use multiprocessing to process different programs in parallel
    print(f"Starting processing with {len(program_tasks)} parallel program workers")
    with Pool() as pool:
        results = pool.map(process_program_sequentially, pool_input)
    
    # Flatten the results and filter out None values
    errors = [err for sublist in results for err in sublist if err]
    
    # Print a summary of errors at the end
    if errors:
        print("\nErrors encountered during mutation testing:")
        for error in errors:
            print(error)
        print(f"\nTotal errors: {len(errors)} out of {total_tasks} tasks")
    else:
        print(f"\nAll {total_tasks} mutation tests completed successfully!")


if __name__ == "__main__":
    main()

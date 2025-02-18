#!/usr/bin/env python3
import os
import submitit
from typing import Tuple, List

# Program names and their corresponding job IDs from test_load_mutations.py
PROGRAM_CONFIGS: List[Tuple[str, str]] = [
    ("sort", "f35ba7e838874bba8335cd9ca5db2aa7"),
    ("reverse", "9db50f10314547858e52a5aff4bc2be4"),
    ("hist", "45728e11fb1043829d4057c016b549b9"),
    ("most_freq", "14ac16fe5f49412aa1ee30461b5769a0"),
    ("shuffle_dyck", "8940b2d2299f45c08b474d151c6d760b"),
    ("shuffle_dyck2", "739a764b78784fc3b4b6f80006eac399"),
]

def get_executor() -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="RASPING-MUTATIONS",
        nodes=1,
        timeout_min=24 * 60,  # 24 hours
        slurm_additional_parameters={
            "reservation": "1g.10gb",
        },
        mem_gb=10,
        cpus_per_task=1,
        gpus_per_node=1,
    )
    return executor

def main():
    # Create output directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Initialize the executor
    executor = get_executor()

    # Create a list of jobs to run
    jobs = []
    
    for program_name, job_id in PROGRAM_CONFIGS:
        # The model_name is the same as program_name for our datasets
        model_name = program_name
        
        # Create the job
        job = executor.submit(
            train_mutated_model,
            model_name=model_name,
            program_name=program_name,
            job_id=job_id,
            max_len=10,
            n_epochs=50000,
            batch_size=256,
            learning_rate=1e-04,
            output_dir=f"saved_data/{program_name}/job_{job_id}/"
        )
        jobs.append(job)
    
    print(f"Submitted {len(jobs)} jobs")

if __name__ == "__main__":
    # Import train_mutated_model here to avoid importing before submitit 
    # has a chance to pickle the function
    from train_mutations import train_mutated_model
    main()
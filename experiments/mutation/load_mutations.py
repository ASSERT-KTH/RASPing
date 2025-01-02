import importlib.util
import sys
from types import ModuleType
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd

module_paths = [
    str(Path(Path(__file__).parent.resolve(), "..", "..").resolve().absolute())
]
if module_paths not in sys.path:
    sys.path.extend(module_paths)

from tracr.rasp import rasp
#from src.functions import generateData, getAcceptedNamesAndInput
def getAcceptedNamesAndInput():
    return {"reverse": ["a","b","c","d","e"], #Tokens doesn't matter much. Only the quantity influnce the results due to encoding (I think)
            "hist": ["a","b","c","d"], #Tokens doesn't matter much. Only the quantity influnce the results due to encoding (I think)
            "sort": [1,2,3,4,5,6], #[0,1,2,3,4,5,6]    Seems to fail sometimes if 0 is included (irrespktive of if 0 is in the failed input or not, don't know why)
            "most-freq": [1,2,3,4,5],
            "shuffle_dyck1": ["(",")"],
            "shuffle_dyck2": ["(",")","{","}"]}
from src.model import Model


def create_module_from_source(source_code: str, module_name: str) -> ModuleType:
    """Create a module from source code string."""
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    exec(source_code, module.__dict__)
    return module


def load_mutation(
    mutation_path: str = "results/aggregated_mutations.json",
    program_name: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load mutations from DataFrame and filter by program name and/or job ID.

    Args:
        mutation_path: Path to the mutations JSON file
        program_name: If provided, only return mutations for this program
        job_id: If provided, only return mutations with this job ID
    """
    df = pd.read_json(mutation_path)

    # Filter for BUGGY_MODEL mutations
    buggy_mutations = {}
    for idx, row in df.iterrows():
        if row["execution_result"].get("status") != "BUGGY_MODEL":
            continue

        # Apply filters if specified
        if program_name and row["program_name"] != program_name:
            continue
        if job_id and row["job_id"] != job_id:
            continue

        buggy_mutations[idx] = row.to_dict()

    return buggy_mutations


def create_model_from_mutation(
    mutation: Dict[str, Any], max_length: int
) -> Optional[Any]:
    """Create a Model object from a mutation entry."""
    try:
        # Create a module from the source code
        source_code = mutation["program_source_after"]
        program_name = mutation["program_name"]
        job_id = mutation["job_id"]
        module_name = f"mutation_{program_name}_{job_id}"

        # Create module from source
        module = create_module_from_source(source_code, module_name)

        # Get accepted inputs for this program
        # FIXME: The key for most_freq is most-freq, not most_freq
        # FIXME: The key for shuffle_dyck is shuffle_dyck1
        program_name_key = program_name
        if program_name == "most_freq":
            program_name_key = "most-freq"
        elif program_name == "shuffle_dyck":
            program_name_key = "shuffle_dyck1"
        accepted_inputs = getAcceptedNamesAndInput()[program_name_key]
        inputs = {t for t in accepted_inputs}

        # Create program
        if program_name == "hist":
            program = module.make_hist()
        elif program_name == "sort":
            program = module.make_sort(
                rasp.tokens, rasp.tokens, max_seq_len=max_length, min_key=min(inputs)
            )
        elif program_name == "reverse":
            program = module.make_reverse(rasp.tokens)
        elif program_name == "most_freq":
            program = module.make_sort_freq(max_length)
        elif program_name == "shuffle_dyck":
            program = module.make_shuffle_dyck(["()"])
        elif program_name == "shuffle_dyck2":
            program = module.make_shuffle_dyck2()
        else:
            raise NotImplementedError(f"Program {program_name} not implemented")

        return Model(program, inputs, max_length, program_name)
    except Exception as e:
        print(
            f"Error creating model from mutation ({mutation['program_name']},{mutation['job_id']})"
        )
        raise e


def load_buggy_models(
    max_length: int,
    mutation_path: str = "results/aggregated_mutations.json",
    program_name: Optional[str] = None,
    job_id: Optional[str] = None,
):
    """Load buggy models from mutations file with optional filtering.

    Args:
        max_length: Maximum sequence length for the models
        mutation_path: Path to the mutations JSON file
        program_name: If provided, only load models for this program
        job_id: If provided, only load models with this job ID
    """
    mutations = load_mutation(mutation_path, program_name, job_id)
    models = {}

    for key, mutation in mutations.items():
        model = create_model_from_mutation(mutation, max_length)
        if model is not None:
            models[key] = model

    return models

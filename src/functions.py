import math
import jax.numpy as jnp
import numpy as np
import pandas as pd
import json
import random
from pathlib import Path

from tracr.compiler import lib
from tracr.rasp import rasp

# Change relative import to absolute
from experiments.mutation.load_mutations import load_buggy_models

from src.model import Model
from src.generators import GENERATORS
import time


# Return all the accepted model names and their corresponding accepted inputs
def getAcceptedNamesAndInput():
    return {
        "reverse": [
            "a",
            "b",
            "c",
            "d",
            "e",
        ],  # Tokens doesn't matter much. Only the quantity influnce the results due to encoding (I think)
        "hist": [
            "a",
            "b",
            "c",
            "d",
        ],  # Tokens doesn't matter much. Only the quantity influnce the results due to encoding (I think)
        "sort": [
            1,
            2,
            3,
            4,
            5,
            6,
        ],  # [0,1,2,3,4,5,6]    Seems to fail sometimes if 0 is included (irrespktive of if 0 is in the failed input or not, don't know why)
        "most-freq": [1, 2, 3, 4, 5],
        "shuffle_dyck1": ["(", ")"],
        "shuffle_dyck2": ["(", ")", "{", "}"],
    }


# Generate a data set based on "name" with "size" samples and a max sequence length of "maxSeqLength"
def generateData(
    name: str,
    maxSeqLength: int,
    size: int = math.inf,
    removeDuplicates=False,
    timeout=10,
    exhaustive=False,
):
    acceptedNamesAndInput = getAcceptedNamesAndInput()

    if name not in acceptedNamesAndInput:
        raise Exception(
            f"{name} is not an accepted name. The accepted names are {acceptedNamesAndInput}"
        )
        return None

    if exhaustive:
        generator = GENERATORS[f"{name}_exhaustive"]
        acceptedTokens = acceptedNamesAndInput[name]
        data = generator(acceptedTokens, maxSeqLength)
        return data[:size] if size < len(data) else data

    generator = GENERATORS[name]
    acceptedTokens = acceptedNamesAndInput[name]

    data = []
    unique_inputs = {}
    start_time = time.time()

    while len(data) < size:
        if time.time() - start_time > timeout:
            print(
                f"Warning: Timeout reached. Only generated {len(data)} unique samples."
            )
            break

        sample = generator(acceptedTokens, maxSeqLength)

        if removeDuplicates:
            # Check if this input sequence is unique
            current_dict = unique_inputs
            for token in sample[0]:
                if token not in current_dict:
                    current_dict[token] = {}
                current_dict = current_dict[token]

            if "@end" not in current_dict:
                current_dict["@end"] = True
                data.append(sample)
        else:
            data.append(sample)

        if len(data) >= size:
            break

    return data


# Generate a rasp model based on "name" and max sequence length "maxLength"
def generateModel(name: str, maxLength: int) -> Model:
    model = None
    acceptedNamesAndInput = getAcceptedNamesAndInput()

    match name:
        case "reverse":
            inputs = {t for t in acceptedNamesAndInput[name]}
            model = Model(lib.make_reverse(rasp.tokens), inputs, maxLength, name)

        case "hist":
            inputs = {t for t in acceptedNamesAndInput[name]}
            model = Model(lib.make_hist(), inputs, maxLength, name)

        case "sort":
            inputs = {t for t in acceptedNamesAndInput[name]}
            model = Model(
                lib.make_sort(
                    rasp.tokens, rasp.tokens, max_seq_len=maxLength, min_key=min(inputs)
                ),
                inputs,
                maxLength,
                name,
            )

        # HACK: We need to canonicalize the program names
        case "most-freq" | "most_freq":
            inputs = {t for t in acceptedNamesAndInput["most-freq"]}
            model = Model(lib.make_sort_freq(maxLength), inputs, maxLength, "most-freq")

        case "shuffle_dyck" | "shuffle_dyck1":
            inputs = {t for t in acceptedNamesAndInput["shuffle_dyck1"]}
            model = Model(
                lib.make_shuffle_dyck(["()"]), inputs, maxLength, "shuffle_dyck1"
            )

        case "shuffle_dyck2":
            inputs = {t for t in acceptedNamesAndInput[name]}
            model = Model(lib.make_shuffle_dyck(["()", "{}"]), inputs, maxLength, name)

        case _:
            print(
                name,
                "is not an accepted name the accepted names are",
                acceptedNamesAndInput,
            )
            return None

    return model


def getMutationJobIDs(name: str, mutationPath: str):
    df = pd.read_json(mutationPath)

    # Filter for BUGGY_MODEL mutations
    jobIDs = []
    for idx, row in df.iterrows():
        if row["execution_result"].get("status") != "BUGGY_MODEL":
            continue
        if row["program_name"] != name:
            continue

        jobIDs.append(row["job_id"])

    return jobIDs


def getMutatedModel(
    name: str, maxLength: int, modelIndex: int, mutationPath: str
) -> Model:
    mutatedNameKey = name
    if mutatedNameKey == "most-freq":
        mutatedNameKey = "most_freq"
    elif mutatedNameKey == "shuffle_dyck1":
        mutatedNameKey = "shuffle_dyck"
    jobIDs = getMutationJobIDs(mutatedNameKey, mutationPath)
    if modelIndex >= len(jobIDs) - 1:
        print(
            "Warning: modelIndex (%d) is larger than available mutations (%d) for model %s"
            % (modelIndex, len(jobIDs), name)
        )
        exit(0)

    models = load_buggy_models(
        mutation_path=mutationPath,
        max_length=maxLength,
        program_name=mutatedNameKey,
        job_id=jobIDs[modelIndex],
    )
    model = list(models.values())[0]
    return model


def make_sort_unique_buggy(vals: rasp.SOp, keys: rasp.SOp) -> rasp.SOp:
    """Returns vals sorted by < relation on keys.

    Only supports unique keys.

    Example usage:
      sort = make_sort(rasp.tokens, rasp.tokens)
      sort([2, 4, 3, 1])
      >> [1, 2, 3, 4]

    Args:
      vals: Values to sort.
      keys: Keys for sorting.
    """
    # BUG: GT instead of LT, resulting in descending order instead of ascending
    smaller = rasp.Select(keys, keys, rasp.Comparison.GT).named("smaller")
    target_pos = rasp.SelectorWidth(smaller).named("target_pos")
    sel_new = rasp.Select(target_pos, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(sel_new, vals).named("sort")


def make_sort_buggy(
    vals: rasp.SOp, keys: rasp.SOp, *, max_seq_len: int, min_key: float
) -> rasp.SOp:
    """Returns vals sorted by < relation on keys, which don't need to be unique.

    The implementation differs from the RASP paper, as it avoids using
    compositions of selectors to break ties. Instead, it uses the arguments
    max_seq_len and min_key to ensure the keys are unique.

    Note that this approach only works for numerical keys.

    Example usage:
      sort = make_sort(rasp.tokens, rasp.tokens, 5, 1)
      sort([2, 4, 3, 1])
      >> [1, 2, 3, 4]
      sort([2, 4, 1, 2])
      >> [1, 2, 2, 4]

    Args:
      vals: Values to sort.
      keys: Keys for sorting.
      max_seq_len: Maximum sequence length (used to ensure keys are unique)
      min_key: Minimum key value (used to ensure keys are unique)

    Returns:
      Output SOp of sort program.
    """
    keys = rasp.SequenceMap(
        lambda x, i: x + min_key * i / max_seq_len, keys, rasp.indices
    )
    return make_sort_unique_buggy(vals, keys)


def generateBuggyModel(name: str, maxLength: int) -> Model:
    acceptedNamesAndInput = getAcceptedNamesAndInput()

    match name:
        case "sort":
            inputs = {t for t in acceptedNamesAndInput[name]}
            return Model(
                make_sort_buggy(
                    rasp.tokens, rasp.tokens, max_seq_len=maxLength, min_key=min(inputs)
                ),
                inputs,
                maxLength,
                name,
            )

        case _:
            print(
                name,
                "is not an accepted name the accepted names are",
                acceptedNamesAndInput,
            )
            return None


# Encode and the data for training based on the target function's rasp encoding
def encodeAndPadData(data, raspFunction: rasp.SOp, acceptedInputs, maxSeqLength: int):
    model = Model(raspFunction, acceptedInputs, maxSeqLength, "Temporary model")
    inputEncoder = model.model.input_encoder
    outputEncoder = model.model.output_encoder

    fillerToken = next(
        iter(outputEncoder.encoding_map)
    )  # A token accepted by the output encoder. Should not have an effect on loss

    X = []
    Y = []

    for inputSeq, outputSeq in data:
        # Copy sequences to avoid overwriting originals and pad them to max length
        x = []
        y = []
        for i in range(maxSeqLength + 1):
            if i < len(inputSeq):  # Assumes that input is same size as output
                x.append(inputSeq[i])
                y.append(outputSeq[i])
            else:
                x.append("compiler_pad")
                y.append(fillerToken)

        y[0] = fillerToken

        X.append(jnp.array(inputEncoder.encode(x)))
        Y.append(jnp.array(outputEncoder.encode(y)))

    X = jnp.array(X)
    Y = jnp.array(Y)

    return X, Y

    # NOTE The first token in the output is ignored but needs to be part of the input since the encoder breaks otherwise


def generateAndEncodeData(name: str, maxLength: int, size: int, removeDuplicates=True):
    data = generateData(name, maxLength, size, removeDuplicates)
    model = generateModel(name, maxLength)
    X, Y = encodeAndPadData(data, model.raspFunction, model.inputs, maxLength)
    return X, Y


# Prints some statistics on the generated dyck data
def checkDyckBalance(data):
    oddLength = 0
    balanced = 0

    for input, output in data:
        if len(input) % 2 == 0:  # length + bos
            oddLength += 1
        if output[1] == 1:
            balanced += 1

    oddLength /= len(data) / 100
    balanced /= len(data) / 100

    print("Percentage of data which is:")
    print("Of odd length:", oddLength)
    print("Balanced:", balanced)


def load_dataset(data_dir: str | Path, program_name: str, split_name: str = "train"):
    """Load a dataset from a JSONL file.

    Args:
        data_dir: Directory containing the dataset files
        program_name: Name of the program (e.g., 'reverse', 'sort')
        split_name: Which split to load ('train', 'val', or 'test')

    Returns:
        List of (input_seq, output_seq) tuples
    """
    data_path = Path(data_dir) / f"{program_name}_{split_name}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    data = []
    with open(data_path) as f:
        for line in f:
            sample = json.loads(line)
            data.append((sample["input"], sample["output"]))
    return data


def split_train_val_test(data, train_pct=0.8, val_pct=0.1):
    """Split data into train, validation and test sets.

    Args:
        data: List of data samples
        train_pct: Percentage of data for training (default: 0.8)
        val_pct: Percentage of data for validation (default: 0.1)

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Convert to list and shuffle deterministically
    random.seed(42)
    data = list(data)
    random.shuffle(data)

    # Calculate split indices
    total = len(data)
    train_idx = int(train_pct * total)
    val_idx = int((train_pct + val_pct) * total)

    return (data[:train_idx], data[train_idx:val_idx], data[val_idx:])


def save_dataset(data_dir: str | Path, program_name: str, split_name: str, data):
    """Save dataset samples to a JSONL file.

    Args:
        data_dir: Directory containing the dataset files
        program_name: Name of the program (e.g., 'reverse', 'sort')
        split_name: Which split to save ('train', 'val', or 'test')
        data: List of (input_seq, output_seq) tuples
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / f"{program_name}_{split_name}.jsonl"
    with open(output_path, "w") as f:
        for input_seq, output_seq in data:
            entry = {"input": input_seq, "output": output_seq}
            json.dump(entry, f, cls=CustomJSONEncoder)
            f.write("\n")


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)

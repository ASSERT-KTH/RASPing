import math
import jax.numpy as jnp
import pandas as pd
import json
from pathlib import Path

from tracr.compiler import lib
from tracr.rasp import rasp

# Change relative import to absolute
from experiments.mutation.load_mutations import load_buggy_models

from src.model import Model
from src.generators import GENERATORS
import time

# Default cache directory
CACHE_DIR = Path("data/exhaustive")


def set_cache_dir(path: str | Path) -> None:
    """Set the directory to use for caching exhaustive data."""
    global CACHE_DIR
    CACHE_DIR = Path(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(name: str, maxSeqLength: int) -> Path:
    """Get the path for cached exhaustive data."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_maxlen{maxSeqLength}.jsonl"


def _save_exhaustive_data(data, cache_path: Path):
    """Save exhaustive data to cache in JSONL format."""
    with open(cache_path, "w") as f:
        for pair in data:
            # Convert input/output sequences to JSON-serializable format
            entry = {"input": pair[0], "output": pair[1]}
            f.write(json.dumps(entry) + "\n")


def _load_exhaustive_data(cache_path: Path):
    """Load exhaustive data from JSONL cache."""
    data = []
    with open(cache_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            # Convert back to tuple format
            data.append((entry["input"], entry["output"]))
    return data


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
        cache_path = _get_cache_path(name, maxSeqLength)
        if cache_path.exists():
            data = _load_exhaustive_data(cache_path)
            return data[:size] if size < len(data) else data

        generator = GENERATORS[f"{name}_exhaustive"]
        acceptedTokens = acceptedNamesAndInput[name]
        data = generator(acceptedTokens, maxSeqLength)
        _save_exhaustive_data(data, cache_path)
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

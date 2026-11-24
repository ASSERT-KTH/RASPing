"""Input-output specifications: generate, split, save, load and encode.

A specification is a list of `(input_seq, output_seq)` pairs, both starting with "BOS".
On disk they live in `data/<data_name>_<split>.jsonl` (`data_name` from `programs`).
"""

from __future__ import annotations

import gzip
import json
import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import jax.numpy as jnp
import numpy as np

from rasping import programs
from rasping.config import DATA_DIR
from rasping.generators import get_generator

if TYPE_CHECKING:
    from rasping.model import Model

Example = tuple[list, list]

# --------------------------------------------------------------------------- jsonl


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def stream_jsonl(path: str | Path) -> Iterable[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as fp:
        for r in records:
            fp.write(json.dumps(r, cls=NumpyJSONEncoder) + "\n")


# --------------------------------------------------------------------------- generation


def generate(
    program: str,
    max_len: int,
    size: float = math.inf,
    remove_duplicates: bool = False,
    timeout: float = 10,
    exhaustive: bool = False,
) -> list[Example]:
    """Generate up to `size` examples of `program` with inputs of length 2..max_len.

    Random generation uses NumPy's global RNG. Exhaustive generation enumerates inputs
    in order of length and truncates to `size` *without shuffling* (F1).
    """
    vocab = list(programs.get(program).vocab)
    if exhaustive:
        data = get_generator(program, exhaustive=True)(vocab, max_len)
        return data[:size] if size < len(data) else data

    generator = get_generator(program)
    data: list[Example] = []
    seen: set[tuple] = set()
    start = time.time()
    while len(data) < size:
        if time.time() - start > timeout:
            print(f"Warning: timeout reached; generated {len(data)} unique samples.")
            break
        sample = generator(vocab, max_len)
        if remove_duplicates:
            key = tuple(sample[0])
            if key in seen:
                continue
            seen.add(key)
        data.append(sample)
    return data


def split_train_val_test(data: list, train_pct: float = 0.8, val_pct: float = 0.1):
    """Deterministic shuffle (seed 42) then split into train / val / test."""
    random.seed(42)
    data = list(data)
    random.shuffle(data)
    n = len(data)
    i, j = int(train_pct * n), int((train_pct + val_pct) * n)
    return data[:i], data[i:j], data[j:]


# --------------------------------------------------------------------------- disk


def dataset_path(program: str, split: str, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / f"{programs.get(program).data_name}_{split}.jsonl"


def load(program: str, split: str = "train", data_dir: Path = DATA_DIR) -> list[Example]:
    path = dataset_path(program, split, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return [(r["input"], r["output"]) for r in stream_jsonl(path)]


def save(program: str, split: str, data: list[Example], data_dir: Path = DATA_DIR) -> Path:
    path = dataset_path(program, split, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, ({"input": x, "output": y} for x, y in data))
    return path


# --------------------------------------------------------------------------- encoding


def encode(data: list[Example], model: "Model") -> tuple[jnp.ndarray, jnp.ndarray]:
    """Encode and pad a specification with `model`'s input/output encoders.

    Inputs are padded with `compiler_pad` to `max_len + 1` (BOS included). Output
    positions that do not count (BOS and padding) are filled with an arbitrary token
    of the output vocabulary; losses and accuracies mask them out.

    Raises KeyError if an output token is not in the model's output vocabulary (the
    compiled architecture cannot represent the target).
    """
    in_enc = model.assembled.input_encoder
    out_enc = model.assembled.output_encoder
    filler = next(iter(out_enc.encoding_map))
    width = model.max_len + 1

    X, Y = [], []
    for inp, out in data:
        x = list(inp[:width]) + ["compiler_pad"] * (width - len(inp))
        y = list(out[:width]) + [filler] * (width - len(out))
        y[0] = filler
        X.append(jnp.array(in_enc.encode(x)))
        Y.append(jnp.array(out_enc.encode(y)))
    return jnp.array(X), jnp.array(Y)

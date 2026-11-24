"""A RASP program compiled by Tracr into a Transformer, and helpers around it."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tracr.compiler import compiling
from tracr.rasp import rasp

from rasping.metrics import exact_match_accuracy

#: Init schemes for `Model.set_random_weights` (fan-in/fan-out as in Haiku's VarianceScaling).
INIT_SCHEMES = ("lecun", "xavier", "kaiming", "unit-normal")


class Model:
    """A Tracr-compiled RASP program.

    Attributes:
        sop:       the RASP program
        vocab:     input vocabulary (set)
        max_len:   maximum input length (BOS excluded)
        name:      program name
        assembled: Tracr's `AssembledTransformerModel` (holds `params` and encoders)
    """

    def __init__(self, sop: rasp.SOp, vocab: Iterable, max_len: int, name: str):
        self.sop = sop
        self.vocab = set(vocab)
        self.max_len = max_len
        self.name = name
        self.assembled = compiling.compile_rasp_to_model(
            sop, self.vocab, max_len, compiler_bos="BOS", mlp_exactness=120
        )
        self._prng_key = jax.random.key(666)
        self.forward_fn = hk.without_apply_rng(hk.transform(self._forward))
        self._jit_accuracy = jax.jit(self._accuracy)

    # ----------------------------------------------------------------- forward pass

    def _forward(self, x):
        compiled = self.assembled.get_compiled_model()
        compiled.use_unembed_argmax = False
        compiled.pad_token = self.pad_token
        return compiled(x, use_dropout=False)

    def forward(self, x, params=None):
        """Model output for encoded inputs `x` (uses the current params by default)."""
        return self.forward_fn.apply(self.params if params is None else params, x)

    @property
    def params(self) -> hk.Params:
        return self.assembled.params

    @property
    def pad_token(self) -> int:
        return self.assembled.input_encoder.encoding_map["compiler_pad"]

    @property
    def n_params(self) -> int:
        return int(sum(np.asarray(t).size for layer in self.params.values() for t in layer.values()))

    # ----------------------------------------------------------------- weights

    def set_prng_key(self, seed: int) -> None:
        self._prng_key = jax.random.key(seed)

    def set_random_weights(self, scheme: str = "xavier", mean: float = 0.0, std: float = 1.0) -> None:
        """Re-initialize all weights, keeping the compiled architecture.

        Fan-scaled schemes draw each >=2-D tensor from N(0, s^2) and zero the biases:
          "lecun"   s^2 = 1 / fan_in            (Haiku's hk.Linear default)
          "xavier"  s^2 = 2 / (fan_in + fan_out) (Glorot & Bengio 2010)
          "kaiming" s^2 = 2 / fan_in             (He et al. 2015)
        with fan_in = prod(shape[:-1]) and fan_out = shape[-1].
        "unit-normal" draws every parameter from N(mean, std^2); not fan-scaled, it
        starts these models numerically broken (kept for the historical baseline).
        """
        self._prng_key, key = jax.random.split(self._prng_key)
        seq = hk.PRNGSequence(key)

        if scheme == "unit-normal":
            new = jax.tree_util.tree_map(
                lambda p: jax.random.normal(next(seq), p.shape) * std + mean, self.params
            )
        elif scheme in ("lecun", "xavier", "kaiming"):

            def init_one(p):
                if len(p.shape) < 2:
                    return jnp.zeros(p.shape, dtype=p.dtype)
                fan_in, fan_out = int(np.prod(p.shape[:-1])), int(p.shape[-1])
                var = {"lecun": 1.0 / fan_in, "xavier": 2.0 / (fan_in + fan_out),
                       "kaiming": 2.0 / fan_in}[scheme]
                return jax.random.normal(next(seq), p.shape, dtype=p.dtype) * np.sqrt(var)

            new = jax.tree_util.tree_map(init_one, self.params)
        else:
            raise ValueError(f"Unknown init scheme {scheme!r}; choose from {INIT_SCHEMES}")
        self.assembled.params = new

    def set_weights(self, params: hk.Params) -> None:
        """Replace the weights; structure and shapes must match the compiled model."""
        cur = self.params
        assert set(params) == set(cur), f"Parameter structure mismatch: {set(params)} vs {set(cur)}"
        for k in params:
            assert set(params[k]) == set(cur[k]), f"Sub-parameter mismatch in {k}"
            for s in params[k]:
                assert params[k][s].shape == cur[k][s].shape, f"Shape mismatch for {k}/{s}"
        self.assembled.params = params

    def save(self, path: str | Path) -> Path:
        """Save params to `path` (a directory gets `model.npy`)."""
        path = Path(path)
        if path.suffix != ".npy":
            path.mkdir(parents=True, exist_ok=True)
            path = path / "model.npy"
        np.save(path, self.params)
        return path

    def load(self, path: str | Path) -> None:
        """Load params saved by `save` (or the trainer) from a file or run directory."""
        path = Path(path)
        if path.is_dir():
            path = path / "model.npy"
        self.set_weights(np.load(path, allow_pickle=True).item())

    # ----------------------------------------------------------------- evaluation

    def apply(self, inp: list) -> list:
        """Decoded output for one raw input sequence (with BOS)."""
        return self.assembled.apply(inp).decoded

    def evaluate(self, data: list[tuple[list, list]]) -> np.ndarray:
        """Per-example exact-match correctness (bool array), decoding one example at a time.

        Slow but independent of the encoding path; compares every position, BOS included.
        """
        out = np.zeros(len(data), dtype=bool)
        for i, (inp, target) in enumerate(data):
            pred = self.apply(inp)
            out[i] = all(pred[j] == target[j] for j in range(len(target)))
        return out

    def _accuracy(self, params, x, y):
        logits = self.forward_fn.apply(params, jnp.array(x)).unembedded_output
        return exact_match_accuracy(logits, x, y, self.pad_token)

    def accuracy(self, X, Y, params=None) -> float:
        """Exact-sequence-match accuracy on encoded data (BOS and padding masked)."""
        return self._jit_accuracy(self.params if params is None else params, X, Y)

    # ----------------------------------------------------------------- comparison

    def diff(self, other: "Model") -> "ModelDiff":
        """Compare architectures (parameter names) and, if equal, weights."""
        a, b = self.params, other.params
        keys = sorted(set(a) | set(b))
        same, diff, lines = [], [], []
        for k in keys:
            if k in a and k in b:
                same.append(k)
                lines.append(f" {k}")
            else:
                diff.append(k)
                lines.append(f"{'-' if k in a else '+'}{k}")
        if diff:
            return ModelDiff(DiffType.ARCH_DIFF, same, diff, "\n".join(lines))

        same, lines = [], []
        for k in a:
            for s in a[k]:
                if jnp.array_equal(a[k][s], b[k][s]):
                    same.append((k, s))
                    lines.append(f" {k}/{s}")
                else:
                    diff.append((k, s))
                    lines.append(f"±{k}/{s}")
        if diff:
            return ModelDiff(DiffType.WEIGHT_DIFF, same, diff, "\n".join(lines))
        return ModelDiff(DiffType.NO_DIFF, same, diff, None)


class DiffType(Enum):
    ARCH_DIFF = 1
    WEIGHT_DIFF = 2
    NO_DIFF = 3


@dataclass
class ModelDiff:
    diff_type: DiffType
    equal: list = field(default_factory=list)
    diff: list = field(default_factory=list)
    diff_str: str | None = None

    def proportion_diff(self) -> float:
        return len(self.diff) / (len(self.equal) + len(self.diff))

    def to_dict(self) -> dict:
        return {"diff_type": self.diff_type.name, "equal": self.equal, "diff": self.diff,
                "diff_str": self.diff_str}

    def __str__(self) -> str:
        return self.diff_str or ""


def compile_program(program: str, source: str | None = None, max_len: int | None = None) -> Model:
    """Compile a base program (reference source by default) or a mutant's source."""
    from rasping import programs
    from rasping.config import MAX_LEN

    max_len = MAX_LEN if max_len is None else max_len
    p = programs.get(program)
    sop = programs.build_sop(source if source is not None else programs.reference_source(p.name),
                             p.name, max_len)
    return Model(sop, p.vocab, max_len, p.name)

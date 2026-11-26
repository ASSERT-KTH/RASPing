"""Global JAX configuration required for compiled RASP programs.

Imported for its side effects by every rasping module that uses JAX, before using it.

Tracr-compiled models rely on exact arithmetic (MLPs built with large constant
weights, `mlp_exactness=120`); float16/bfloat16 matmuls make the compiled model
disagree with the RASP program, so we enable x64 and force float32 matmuls.

GPU reductions are made deterministic (`config.DETERMINISTIC_XLA_FLAG`, F39): without it,
two identical runs from a compiled start can end at test accuracy 0.01 and 1.0. XLA reads
the flag when its backend starts, so it only applies if JAX has not computed anything
before `rasping` is imported. An explicit setting of the flag in XLA_FLAGS wins.
`repair` records XLA_FLAGS in each run.json.
"""

import os

from rasping.config import DETERMINISTIC_XLA_FLAG

if DETERMINISTIC_XLA_FLAG.split("=")[0] not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {DETERMINISTIC_XLA_FLAG}".strip()

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "float32")
# v1 also enabled jax_debug_nans globally (slow, and it changes behaviour for every
# importer). Divergence is instead detected explicitly by the Trainer (F23).

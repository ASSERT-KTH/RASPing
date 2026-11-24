"""Global JAX configuration required for compiled RASP programs.

Imported for its side effects by every rasping module that uses JAX, before using it.

Tracr-compiled models rely on exact arithmetic (MLPs built with large constant
weights, `mlp_exactness=120`); float16/bfloat16 matmuls make the compiled model
disagree with the RASP program, so we enable x64 and force float32 matmuls.
"""

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "float32")
# v1 also enabled jax_debug_nans globally (slow, and it changes behaviour for every
# importer). Divergence is instead detected explicitly by the Trainer (F23).

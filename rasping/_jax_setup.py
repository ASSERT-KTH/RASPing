"""Global JAX configuration required for compiled RASP programs.

Imported for its side effects by `rasping/__init__.py`, before anything touches JAX.

Tracr-compiled models rely on exact arithmetic (MLPs built with large constant
weights, `mlp_exactness=120`); float16/bfloat16 matmuls make the compiled model
disagree with the RASP program, so we enable x64 and force float32 matmuls.
"""

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "float32")
# Raise on NaNs. Kept from the v1 code for behavioural equivalence; see F23.
jax.config.update("jax_debug_nans", True)

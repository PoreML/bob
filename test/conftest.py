"""Shared test config.

Enable double precision before any bob/jax arrays are created — LBM validation
needs float64 (default float32 would swamp the tolerances below).
"""

import jax

jax.config.update("jax_enable_x64", True)

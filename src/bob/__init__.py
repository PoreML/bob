"""bob — a two-phase (color-gradient) lattice-Boltzmann solver in JAX.

D3Q19 stack: ``bob.d3q19`` (lattice constants) / ``bob.lbm3d`` (single-phase core) /
``bob.color3d`` (two-phase solver) / ``bob.porous3d`` (porous-media drainage).
"""

import os

# On GPU, JAX preallocates 75% of the *total* card memory by default — on a
# shared GPU that instantly OOMs against other jobs. Allocate on demand
# instead (bob's working sets are small: a 256^3 two-phase state is ~5 GB in
# float64). Must be set before the backend initializes, which the float64
# check below triggers; setdefault keeps it user-overridable.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _preload_nvjitlink():
    """A system CUDA on LD_LIBRARY_PATH (e.g. /usr/local/cuda/lib64) can shadow
    the pip-installed libnvJitLink with an older one; cuSPARSE then fails to
    load (undefined symbol __nvJitLinkGetErrorLogSize_12_9) and JAX silently
    falls back to CPU. Preloading the venv's copy wins: the dynamic loader
    reuses an already-loaded soname when cuSPARSE asks for its dependency.
    No-op when the CUDA wheels aren't installed (CPU-only boxes, macOS)."""
    import ctypes
    import glob
    import sysconfig

    pattern = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "nvjitlink", "lib", "libnvJitLink.so.*")
    for path in glob.glob(pattern):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_nvjitlink()

import jax

# Precision: **float32 is the solver default** — LBM is bandwidth-bound, so fp32 is
# ~2x steps/s and half the memory, validated at parity with fp64 on the drainage
# stack. Lattice constants are built at import time, so precision is decided
# HERE, before any bob submodule loads:
#   * BOB_FP32=1  forces float32 globally, overriding a caller's earlier
#     jax.config.update("jax_enable_x64", True) (the demos' fp64 headers);
#   * BOB_FP64=1  opts into float64;
#   * otherwise an explicit x64 enable by the caller (before importing bob) is
#     respected — the test suite and fp64-calibrated demos do this — and plain
#     `import bob` runs float32.
if os.environ.get("BOB_FP32") == "1":
    jax.config.update("jax_enable_x64", False)
elif os.environ.get("BOB_FP64") == "1":
    jax.config.update("jax_enable_x64", True)

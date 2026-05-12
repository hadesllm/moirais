"""ctypes bridge for MORIE engine C kernels.

Loads ``engine_kernels.dylib`` (macOS) or ``engine_kernels.so`` (Linux)
for Accelerate.framework-backed SIMD operations. Falls back to pure
NumPy if the shared library is not available.

Security:
    - Library path is resolved relative to this file only (no PATH search)
    - No user-supplied paths, no eval/exec
    - All C functions validate inputs and return error codes
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Library loading -- same pattern as quant_bridge.py
# ---------------------------------------------------------------------------

_lib: ctypes.CDLL | None = None
_LIB_DIR = Path(__file__).parent

MORIE_OK = 0
MORIE_ERR_NULL = -1
MORIE_ERR_SIZE = -2


def _load_lib() -> ctypes.CDLL | None:
    """Try to load the engine kernels shared library."""
    global _lib
    if _lib is not None:
        return _lib

    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = _LIB_DIR / f"engine_kernels{suffix}"

    if not lib_path.exists():
        return None

    try:
        lib = ctypes.CDLL(str(lib_path))

        # Declare function signatures
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_int_p = ctypes.POINTER(ctypes.c_int)

        lib.morie_rmsnorm.restype = ctypes.c_int
        lib.morie_rmsnorm.argtypes = [c_float_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_float]

        lib.morie_rope.restype = ctypes.c_int
        lib.morie_rope.argtypes = [c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]

        lib.morie_matvec.restype = ctypes.c_int
        lib.morie_matvec.argtypes = [c_float_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_int]

        lib.morie_silu_inplace.restype = ctypes.c_int
        lib.morie_silu_inplace.argtypes = [c_float_p, ctypes.c_int]

        lib.morie_softmax_inplace.restype = ctypes.c_int
        lib.morie_softmax_inplace.argtypes = [c_float_p, ctypes.c_int]

        lib.morie_elemwise_mul.restype = ctypes.c_int
        lib.morie_elemwise_mul.argtypes = [c_float_p, c_float_p, c_float_p, ctypes.c_int]

        lib.morie_dot.restype = ctypes.c_int
        lib.morie_dot.argtypes = [c_float_p, c_float_p, ctypes.c_int, c_float_p]

        lib.morie_argmax.restype = ctypes.c_int
        lib.morie_argmax.argtypes = [c_float_p, ctypes.c_int, c_int_p]

        _lib = lib
        return lib
    except OSError:
        return None


def _as_f32(arr: np.ndarray) -> np.ndarray:
    """Ensure array is contiguous float32."""
    return np.ascontiguousarray(arr, dtype=np.float32)


def _ptr(arr: np.ndarray):
    """Get ctypes float pointer from numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ---------------------------------------------------------------------------
# Public API -- each function tries C, falls back to NumPy
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return True if the C kernel library is loaded."""
    return _load_lib() is not None


def rmsnorm(x: NDArray, weight: NDArray, eps: float = 1e-5) -> NDArray:
    """RMS normalization using C kernel or NumPy fallback."""
    x = _as_f32(x)
    weight = _as_f32(weight)
    lib = _load_lib()

    if lib is not None:
        out = np.empty_like(x)
        rc = lib.morie_rmsnorm(_ptr(x), _ptr(weight), _ptr(out), len(x), ctypes.c_float(eps))
        if rc == MORIE_OK:
            return out

    # NumPy fallback
    rms = np.sqrt(np.mean(x * x) + eps)
    return x / rms * weight


def rope(q: NDArray, k: NDArray, position: int, head_dim: int, freq_base: float = 10000.0) -> tuple[NDArray, NDArray]:
    """Apply RoPE using C kernel or NumPy fallback."""
    q = _as_f32(q).copy()
    k = _as_f32(k).copy()
    lib = _load_lib()

    if lib is not None:
        rc = lib.morie_rope(_ptr(q), _ptr(k), head_dim, position, ctypes.c_float(freq_base))
        if rc == MORIE_OK:
            return q, k

    # NumPy fallback
    half = head_dim // 2
    freqs = 1.0 / (freq_base ** (np.arange(0, half, dtype=np.float32) / half))
    angles = position * freqs
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)
    q0, q1 = q[:half].copy(), q[half:].copy()
    q[:half] = q0 * cos_a - q1 * sin_a
    q[half:] = q0 * sin_a + q1 * cos_a
    k0, k1 = k[:half].copy(), k[half:].copy()
    k[:half] = k0 * cos_a - k1 * sin_a
    k[half:] = k0 * sin_a + k1 * cos_a
    return q, k


def matvec(A: NDArray, x: NDArray) -> NDArray:
    """Matrix-vector multiply using C kernel (Accelerate BLAS) or NumPy."""
    A = _as_f32(A)
    x = _as_f32(x)
    lib = _load_lib()

    if lib is not None:
        rows, cols = A.shape
        out = np.empty(rows, dtype=np.float32)
        rc = lib.morie_matvec(_ptr(A), _ptr(x), _ptr(out), rows, cols)
        if rc == MORIE_OK:
            return out

    return (A @ x).astype(np.float32)


def silu_inplace(x: NDArray) -> NDArray:
    """SiLU activation in-place using C kernel or NumPy."""
    x = _as_f32(x).copy()
    lib = _load_lib()

    if lib is not None:
        rc = lib.morie_silu_inplace(_ptr(x), len(x))
        if rc == MORIE_OK:
            return x

    return x * (1.0 / (1.0 + np.exp(-x)))


def softmax(x: NDArray) -> NDArray:
    """Softmax using C kernel or NumPy."""
    x = _as_f32(x).copy()
    lib = _load_lib()

    if lib is not None:
        rc = lib.morie_softmax_inplace(_ptr(x), len(x))
        if rc == MORIE_OK:
            return x

    x_max = np.max(x)
    e = np.exp(x - x_max)
    return e / e.sum()


def argmax(x: NDArray) -> int:
    """Argmax using C kernel or NumPy."""
    x = _as_f32(x)
    lib = _load_lib()

    if lib is not None:
        result = ctypes.c_int(0)
        rc = lib.morie_argmax(_ptr(x), len(x), ctypes.byref(result))
        if rc == MORIE_OK:
            return result.value

    return int(np.argmax(x))

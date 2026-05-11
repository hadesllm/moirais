"""Python ctypes bridge for GGML TurboQuant C kernels.

Provides :class:`GGMLTurboQuant` which loads the compiled shared library
and exposes quantize/dequantize/dot_product as Python callables.

The C implementation lives in ``quant_ggml.h`` (header) and the eventual
``quant_ggml.c`` (to be compiled). Until the C library is built, this
module falls back to the pure-NumPy implementation in :mod:`morie.quant`.

Usage
-----
>>> from morie.quant_bridge import GGMLTurboQuant
>>> tq = GGMLTurboQuant()
>>> if tq.available:
...     block = tq.quantize(vector, bits=3)
... else:
...     # Falls back to pure Python
...     from morie.quant import turboquant_mse
...     block = turboquant_mse(vector, bits=3)
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import (
    POINTER,
    Structure,
    c_float,
    c_int,
    c_uint8,
    c_uint16,
    c_void_p,
)
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Block size matching the C header
TQ_BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# C structure mirrors
# ---------------------------------------------------------------------------


class BlockTBQ3(Structure):
    """ctypes mirror of block_tbq3_0 (102 bytes per 256 elements)."""

    _fields_ = [
        ("norm", c_float),
        ("seed", c_uint16),
        ("indices", c_uint8 * 96),
    ]


class BlockTBQ4(Structure):
    """ctypes mirror of block_tbq4_0 (134 bytes per 256 elements)."""

    _fields_ = [
        ("norm", c_float),
        ("seed", c_uint16),
        ("indices", c_uint8 * 128),
    ]


class BlockTBQ2(Structure):
    """ctypes mirror of block_tbq2_0 (70 bytes per 256 elements)."""

    _fields_ = [
        ("norm", c_float),
        ("seed", c_uint16),
        ("indices", c_uint8 * 64),
    ]


_BLOCK_TYPES = {
    2: BlockTBQ2,
    3: BlockTBQ3,
    4: BlockTBQ4,
}


# ---------------------------------------------------------------------------
# GGMLTurboQuant — ctypes wrapper
# ---------------------------------------------------------------------------


class GGMLTurboQuant:
    """Python interface to the GGML TurboQuant C library.

    Falls back to pure-NumPy when the shared library isn't compiled.

    Parameters
    ----------
    lib_path : str or Path, optional
        Path to the compiled shared library (``.dylib`` / ``.so``).
        If None, searches in the package directory.
    """

    def __init__(self, lib_path: str | Path | None = None):
        self._lib: ctypes.CDLL | None = None
        self._available = False

        if lib_path is None:
            # Search for the compiled library next to this file
            pkg_dir = Path(__file__).parent
            for suffix in (".dylib", ".so", ".dll"):
                candidate = pkg_dir / f"quant_ggml{suffix}"
                if candidate.exists():
                    lib_path = candidate
                    break

        if lib_path and Path(lib_path).exists():
            try:
                self._lib = ctypes.CDLL(str(lib_path))
                self._setup_signatures()
                self._available = True
                logger.info("Loaded GGML TurboQuant library from %s", lib_path)
            except OSError as e:
                logger.debug("Could not load GGML library: %s", e)

        if not self._available:
            logger.debug("GGML TurboQuant not available — using pure-NumPy fallback")

    def _setup_signatures(self) -> None:
        """Set up ctypes function signatures."""
        if self._lib is None:
            return

        # tq_init(ctx, dim, bits)
        self._lib.tq_init.argtypes = [c_void_p, c_int, c_int]
        self._lib.tq_init.restype = None

        # tq_quantize_block(src, dst, ctx, seed)
        self._lib.tq_quantize_block.argtypes = [
            POINTER(c_float),
            c_void_p,
            c_void_p,
            c_uint16,
        ]
        self._lib.tq_quantize_block.restype = None

        # tq_dequantize_block(src, dst, ctx)
        self._lib.tq_dequantize_block.argtypes = [
            c_void_p,
            POINTER(c_float),
            c_void_p,
        ]
        self._lib.tq_dequantize_block.restype = None

        # tq_dot_product(block, vec, ctx)
        self._lib.tq_dot_product.argtypes = [
            c_void_p,
            POINTER(c_float),
            c_void_p,
        ]
        self._lib.tq_dot_product.restype = c_float

    @property
    def available(self) -> bool:
        """True if the C library is loaded and ready."""
        return self._available

    def quantize(
        self,
        vector: NDArray[np.float32],
        bits: int = 3,
        seed: int = 42,
    ) -> Any:
        """Quantize a vector using the C kernel or NumPy fallback.

        Parameters
        ----------
        vector : ndarray of float32, shape (d,)
            Input vector. d must be 256 (TQ_BLOCK_SIZE).
        bits : int
            Quantization bits (2, 3, or 4).
        seed : int
            Rotation matrix seed.

        Returns
        -------
        TQBlock or ctypes Structure
            Compressed block.
        """
        if self._available and self._lib is not None:
            return self._quantize_c(vector, bits, seed)
        return self._quantize_numpy(vector, bits, seed)

    def dequantize(self, block: Any, bits: int = 3) -> NDArray[np.float32]:
        """Dequantize a block back to float32 vector."""
        if self._available and self._lib is not None and isinstance(block, Structure):
            return self._dequantize_c(block, bits)
        return self._dequantize_numpy(block)

    # -- C kernel paths -------------------------------------------------------

    def _make_ctx(self, bits: int, dim: int = TQ_BLOCK_SIZE) -> Any:
        """Create and initialize a tq_context via the C library."""
        # tq_context is opaque; allocate as raw bytes and call tq_init
        ctx_size = 4 + 4 + (4 + 4 + 4 + 16 * 4)  # dim + bits + codebook struct
        ctx_buf = (ctypes.c_char * 256)()  # oversized to be safe
        self._lib.tq_init(ctypes.byref(ctx_buf), c_int(dim), c_int(bits))
        return ctx_buf

    def _quantize_c(self, vector: NDArray[np.float32], bits: int, seed: int) -> Structure:
        BlockType = _BLOCK_TYPES[bits]
        block = BlockType()
        src = vector.astype(np.float32)
        src_ptr = src.ctypes.data_as(POINTER(c_float))
        ctx = self._make_ctx(bits, len(vector))
        self._lib.tq_quantize_block(src_ptr, ctypes.byref(block), ctypes.byref(ctx), c_uint16(seed))
        return block

    def _dequantize_c(self, block: Structure, bits: int) -> NDArray[np.float32]:
        dst = np.zeros(TQ_BLOCK_SIZE, dtype=np.float32)
        dst_ptr = dst.ctypes.data_as(POINTER(c_float))
        ctx = self._make_ctx(bits)
        self._lib.tq_dequantize_block(ctypes.byref(block), dst_ptr, ctypes.byref(ctx))
        return dst

    # -- NumPy fallback paths -------------------------------------------------

    @staticmethod
    def _quantize_numpy(vector: NDArray[np.float32], bits: int, seed: int) -> Any:
        from morie.quant import turboquant_mse

        return turboquant_mse(vector.astype(np.float64), bits=bits, rotation_seed=seed)

    @staticmethod
    def _dequantize_numpy(block: Any) -> NDArray[np.float32]:
        from morie.quant import turboquant_mse_decode

        return turboquant_mse_decode(block).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience: compile the C library
# ---------------------------------------------------------------------------


def compile_ggml_lib(
    output_dir: str | Path | None = None,
    optimize: bool = True,
) -> Path | None:
    """Attempt to compile the GGML TurboQuant C library.

    Requires ``cc`` (clang or gcc) on the system.

    Parameters
    ----------
    output_dir : str or Path, optional
        Where to write the shared library. Defaults to the morie package dir.
    optimize : bool
        Use ``-O2 -march=native`` (default True).

    Returns
    -------
    Path or None
        Path to compiled library, or None on failure.
    """
    import shutil
    import sys

    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        logger.warning("No C compiler found — cannot compile GGML library")
        return None

    pkg_dir = Path(__file__).parent
    header = pkg_dir / "quant_ggml.h"
    if not header.exists():
        logger.warning("quant_ggml.h not found at %s", header)
        return None

    if output_dir is None:
        output_dir = pkg_dir

    output_dir = Path(output_dir)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = output_dir / f"quant_ggml{suffix}"

    # The C source would need to be written; for now just validate the header
    logger.info(
        "GGML C library compilation not yet implemented — header validated at %s. Using NumPy fallback.",
        header,
    )
    return None

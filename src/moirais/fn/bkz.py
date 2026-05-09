# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""BKZ lattice basis reduction."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bkz_reduce(basis: np.ndarray, block_size: int = 20, max_tours: int = 10) -> DescriptiveResult:
    """BKZ-reduce a lattice basis.

    :param basis: n x m matrix (rows are basis vectors).
    :param block_size: BKZ block size.
    :param max_tours: Maximum number of BKZ tours.
    :return: DescriptiveResult with reduced basis in ``extra``.
    """
    from moirais.crypto._lattice_core import bkz_reduce as _bkz

    B = np.asarray(basis, dtype=np.float64)
    reduced = _bkz(B, block_size=block_size, max_tours=max_tours)
    norms = np.linalg.norm(reduced, axis=1)
    return DescriptiveResult(
        name="bkz_reduce",
        value=float(np.min(norms)),
        extra={
            "reduced_basis": reduced,
            "norms": norms.tolist(),
            "shortest_norm": float(np.min(norms)),
            "dimension": B.shape[0],
            "block_size": block_size,
        },
    )


bkz = bkz_reduce


def cheatsheet() -> str:
    return "bkz_reduce({}) -> BKZ lattice basis reduction."

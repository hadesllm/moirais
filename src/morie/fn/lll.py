# morie.fn -- function file (hadesllm/morie)
"""LLL lattice basis reduction."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lll_reduce(basis: np.ndarray, delta: float = 0.75) -> DescriptiveResult:
    """LLL-reduce a lattice basis.

    :param basis: n x m matrix (rows are basis vectors).
    :param delta: Lovász condition parameter (0.25 < delta <= 1).
    :return: DescriptiveResult with reduced basis in ``extra``.
    """
    from morie.crypto._lattice_core import lll_reduce as _lll

    B = np.asarray(basis, dtype=np.float64)
    reduced = _lll(B, delta=delta)
    norms = np.linalg.norm(reduced, axis=1)
    return DescriptiveResult(
        name="lll_reduce",
        value=float(np.min(norms)),
        extra={
            "reduced_basis": reduced,
            "norms": norms.tolist(),
            "shortest_norm": float(np.min(norms)),
            "dimension": B.shape[0],
            "delta": delta,
        },
    )


lll = lll_reduce


def cheatsheet() -> str:
    return "lll_reduce({}) -> LLL lattice basis reduction."

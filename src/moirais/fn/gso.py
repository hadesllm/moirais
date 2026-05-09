# moirais.fn — function file (hadesllm/moirais)
"""Gram-Schmidt orthogonalization for lattice bases."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gram_schmidt_orth(basis: np.ndarray) -> DescriptiveResult:
    """Gram-Schmidt orthogonalization of a lattice basis.

    :param basis: n x m matrix (rows are basis vectors).
    :return: DescriptiveResult with orthogonal basis and mu coefficients.
    """
    from moirais.crypto._lattice_core import gram_schmidt as _gso

    B = np.asarray(basis, dtype=np.float64)
    B_star, mu = _gso(B)
    norms = np.linalg.norm(B_star, axis=1)
    return DescriptiveResult(
        name="gram_schmidt",
        value=float(np.min(norms)),
        extra={
            "orthogonal_basis": B_star,
            "mu": mu,
            "norms": norms.tolist(),
            "dimension": B.shape[0],
        },
    )


gso = gram_schmidt_orth


def cheatsheet() -> str:
    return "gram_schmidt_orth({}) -> Gram-Schmidt orthogonalization for lattice bases."

# morie.fn — function file (hadesllm/morie)
"""Matrix norms (1, 2, infinity, Frobenius)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def matrix_norms(
    A: np.ndarray,
) -> DescriptiveResult:
    """Compute standard matrix norms.

    Returns the 1-norm, 2-norm (spectral), infinity-norm, and Frobenius norm.

    Parameters
    ----------
    A : ndarray
        Input matrix.

    Returns
    -------
    DescriptiveResult
        ``value`` is the 2-norm; ``extra`` has all four norms.
    """
    A = np.asarray(A, dtype=float)
    n1 = float(np.linalg.norm(A, 1))
    n2 = float(np.linalg.norm(A, 2))
    ninf = float(np.linalg.norm(A, np.inf))
    nfro = float(np.linalg.norm(A, "fro"))
    return DescriptiveResult(
        name="Matrix Norms",
        value=n2,
        extra={"norm_1": n1, "norm_2": n2, "norm_inf": ninf, "norm_fro": nfro},
    )


matnm = matrix_norms

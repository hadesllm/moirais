# morie.fn -- function file (hadesllm/morie)
"""Matrix exponential via scaling and squaring."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def matrix_exp(
    A: np.ndarray,
    *,
    terms: int = 20,
) -> DescriptiveResult:
    """Matrix exponential exp(A) via scaling-and-squaring with Taylor series.

    Scales A down by 2^s so ||A/2^s|| < 1, computes the Taylor series
    for exp(A/2^s), then squares s times.

    Parameters
    ----------
    A : ndarray
        Square matrix (n x n).
    terms : int
        Number of Taylor terms.

    Returns
    -------
    DescriptiveResult
        ``value`` is the trace of exp(A); ``extra`` has the result matrix.
    """
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    norm_a = np.linalg.norm(A, 1)
    s = max(0, int(np.ceil(np.log2(max(norm_a, 1e-15)))) + 1)
    As = A / (2**s)
    F = np.eye(A.shape[0])
    Ak = np.eye(A.shape[0])
    for k in range(1, terms + 1):
        Ak = Ak @ As / k
        F = F + Ak
    for _ in range(s):
        F = F @ F
    return DescriptiveResult(
        name="Matrix Exponential",
        value=float(np.trace(F)),
        extra={"matrix": F},
    )


mtexp = matrix_exp

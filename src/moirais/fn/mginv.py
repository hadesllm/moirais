# moirais.fn — function file (hadesllm/moirais)
"""Moore-Penrose pseudoinverse via SVD."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def pseudoinverse(
    A: np.ndarray,
    *,
    rcond: float = 1e-15,
) -> DescriptiveResult:
    """Moore-Penrose pseudoinverse via SVD.

    Computes A+ such that AA+A = A, A+AA+ = A+, and both AA+ and A+A
    are Hermitian.

    Parameters
    ----------
    A : ndarray
        Input matrix (m x n).
    rcond : float
        Cutoff for small singular values.

    Returns
    -------
    DescriptiveResult
        ``value`` is the numerical rank; ``extra`` has A_pinv and
        singular_values.
    """
    A = np.asarray(A, dtype=float)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    cutoff = rcond * S[0] if len(S) > 0 else 0.0
    mask = cutoff < S
    S_inv = np.where(mask, 1.0 / S, 0.0)
    A_pinv = (Vt.T * S_inv) @ U.T
    rank = int(np.sum(mask))
    return DescriptiveResult(
        name="Pseudoinverse",
        value=rank,
        extra={"A_pinv": A_pinv, "singular_values": S},
    )


mginv = pseudoinverse

"""Total least squares via SVD."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def total_least_squares(
    A: np.ndarray,
    b: np.ndarray,
) -> DescriptiveResult:
    """Total Least Squares (errors-in-variables regression).

    Minimises the Frobenius norm of [dA, db] such that (A+dA)x = b+db,
    solved via the SVD of the augmented matrix [A | b].

    Parameters
    ----------
    A : ndarray
        Design matrix (m x n).
    b : ndarray
        Observation vector of length m.

    Returns
    -------
    DescriptiveResult
        ``value`` is the smallest singular value used; ``extra`` has x.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    C = np.hstack([A, b])
    _, S, Vt = np.linalg.svd(C, full_matrices=True)
    V = Vt.T
    n = A.shape[1]
    V_ab = V[:n, n:]
    V_bb = V[n:, n:]
    if abs(V_bb[0, 0]) < 1e-15:
        raise ValueError("TLS problem is degenerate (V_bb ~ 0)")
    x = (-V_ab / V_bb[0, 0]).ravel()
    return DescriptiveResult(
        name="Total Least Squares",
        value=float(S[-1]),
        extra={"x": x},
    )


trlsq = total_least_squares

# moirais.fn — function file (hadesllm/moirais)
"""Matrix logarithm via inverse scaling and squaring."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def matrix_log(
    A: np.ndarray,
    *,
    maxsqrt: int = 20,
    pade_order: int = 8,
) -> DescriptiveResult:
    """Matrix logarithm log(A) via inverse scaling and squaring.

    Repeatedly takes matrix square roots to bring A close to I, then
    uses a Pade approximant for log(I + X).

    Parameters
    ----------
    A : ndarray
        Square matrix with no eigenvalues on the negative real axis.
    maxsqrt : int
        Maximum number of square-root reductions.
    pade_order : int
        Pade approximation order.

    Returns
    -------
    DescriptiveResult
        ``value`` is the trace of log(A); ``extra`` has the result matrix.
    """
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    n = A.shape[0]
    I = np.eye(n)
    T = A.copy()
    s = 0
    for _ in range(maxsqrt):
        if np.linalg.norm(T - I, 1) < 0.5:
            break
        eigvals = np.linalg.eigvals(T)
        if np.any(np.real(eigvals) <= 0):
            T = np.real(np.linalg.matrix_power(T, 1))
        from scipy.linalg import sqrtm
        T = np.real(sqrtm(T))
        s += 1
    X = T - I
    L = np.zeros_like(X)
    Xk = X.copy()
    for k in range(1, pade_order + 1):
        L += ((-1) ** (k + 1)) * Xk / k
        Xk = Xk @ X
    L = (2**s) * L
    return DescriptiveResult(
        name="Matrix Logarithm",
        value=float(np.trace(L)),
        extra={"matrix": L, "sqrt_reductions": s},
    )


mtlog = matrix_log

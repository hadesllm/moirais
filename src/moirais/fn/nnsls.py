# moirais.fn — function file (hadesllm/moirais)
"""Non-negative least squares (Lawson-Hanson)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def nnls(
    A: np.ndarray,
    b: np.ndarray,
    *,
    maxiter: int = 3000,
) -> DescriptiveResult:
    """Non-negative least squares: min ||Ax - b||_2 subject to x >= 0.

    Implements the active-set method of Lawson & Hanson (1974).

    Parameters
    ----------
    A : ndarray
        Design matrix (m x n).
    b : ndarray
        Observation vector of length m.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the residual norm; ``extra`` has x.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape
    x = np.zeros(n)
    P = set()
    Z = set(range(n))
    w = A.T @ (b - A @ x)
    for _ in range(maxiter):
        if not Z or np.max(w[list(Z)]) <= 0:
            break
        t = list(Z)[np.argmax(w[list(Z)])]
        P.add(t)
        Z.discard(t)
        while True:
            idx = sorted(P)
            Ap = A[:, idx]
            sp = np.linalg.lstsq(Ap, b, rcond=None)[0]
            if np.all(sp > 0):
                x_new = np.zeros(n)
                for i, j in enumerate(idx):
                    x_new[j] = sp[i]
                x = x_new
                break
            neg = [idx[i] for i in range(len(idx)) if sp[i] <= 0]
            alpha = min(x[j] / (x[j] - sp[list(sorted(P)).index(j)] + 1e-30) for j in neg)
            x_vec = np.zeros(n)
            for i, j in enumerate(idx):
                x_vec[j] = sp[i]
            x = x + alpha * (x_vec - x)
            for j in list(P):
                if abs(x[j]) < 1e-15:
                    P.discard(j)
                    Z.add(j)
        w = A.T @ (b - A @ x)
    residual = float(np.linalg.norm(A @ x - b))
    return DescriptiveResult(name="NNLS", value=residual, extra={"x": x})


nnsls = nnls

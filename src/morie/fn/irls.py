# morie.fn — function file (hadesllm/morie)
"""Iteratively reweighted least squares for Lp minimization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "So this is how liberty dies... with thunderous applause."


def iteratively_reweighted_ls(A, b, p: float = 1.0, n_iter: int = 50, eps: float = 1e-6, **kwargs) -> DescriptiveResult:
    """Iteratively reweighted least squares (IRLS) for Lp minimization.

    Solves: min_x  ||x||_p  subject to  Ax = b

    Parameters
    ----------
    A : array-like, shape (m, n)
        Constraint matrix (m < n for underdetermined system).
    b : array-like, shape (m,)
        Observation vector.
    p : float
        Lp norm exponent, 0 < p <= 2 (default 1.0).
    n_iter : int
        Number of IRLS iterations (default 50).
    eps : float
        Regularization to avoid division by zero (default 1e-6).

    Returns
    -------
    DescriptiveResult
        ``value`` is final Lp norm; ``extra`` has ``x`` (solution),
        ``weights``, ``iterations``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    for it in range(n_iter):
        w = np.power(np.abs(x) + eps, (p - 2.0) / 2.0)
        W = np.diag(w**2)
        AW = A @ W @ A.T
        lam = np.linalg.solve(AW, b)
        x_new = W @ A.T @ lam
        if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-15) < 1e-8:
            x = x_new
            break
        x = x_new

    lp_norm = float(np.sum(np.abs(x) ** p))
    return DescriptiveResult(
        name="iteratively_reweighted_ls",
        value=lp_norm,
        extra={"x": x, "weights": w, "iterations": it + 1},
    )


irls = iteratively_reweighted_ls


def cheatsheet() -> str:
    return "iteratively_reweighted_ls({}) -> Iteratively reweighted least squares for Lp minimization."

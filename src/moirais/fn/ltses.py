# moirais.fn — function file (hadesllm/moirais)
"""Least trimmed squares (LTS) regression."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def least_trimmed_squares(
    X,
    y,
    *,
    alpha: float = 0.5,
    n_subsets: int = 500,
    seed: int = 42,
) -> ESRes:
    """Least trimmed squares regression (Rousseeuw, 1984).

    Minimises the sum of the h smallest squared residuals.

    Parameters
    ----------
    X : array-like, shape (n, p) or (n,)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    alpha : float
        Coverage fraction (default 0.5 = 50 % breakdown).
    n_subsets : int
        Number of random subsets to try.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    ESRes
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    h = int(np.ceil(n * alpha)) + p
    h = max(h, p + 1)
    h = min(h, n)
    if n < p + 1:
        raise ValueError("Need n > p observations.")

    rng = np.random.default_rng(seed)
    best_obj = np.inf
    best_beta = np.zeros(p)

    for _ in range(n_subsets):
        idx = rng.choice(n, size=p + 1, replace=False)
        try:
            beta = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        resid2 = (y - X @ beta) ** 2
        order = np.argsort(resid2)
        obj = float(np.sum(resid2[order[:h]]))
        if obj < best_obj:
            best_obj = obj
            best_beta = beta

    resid = y - X @ best_beta
    s = 1.4826 * np.median(np.abs(resid))
    return ESRes(
        measure="lts",
        estimate=best_obj,
        n=n,
        extra={
            "coefficients": best_beta.tolist(),
            "scale": float(s),
            "h": h,
            "alpha": alpha,
        },
    )


ltses = least_trimmed_squares


def cheatsheet() -> str:
    return "least_trimmed_squares(X, y) -> LTS robust regression."

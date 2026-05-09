# moirais.fn — function file (hadesllm/moirais)
"""Least median of squares (LMS) regression."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def least_median_squares(
    X,
    y,
    *,
    n_subsets: int = 500,
    seed: int = 42,
) -> ESRes:
    """Least median of squares regression (Rousseeuw, 1984).

    Minimises the median of squared residuals.

    Parameters
    ----------
    X : array-like, shape (n, p) or (n,)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    n_subsets : int
        Number of random subsets.
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
        obj = float(np.median(resid2))
        if obj < best_obj:
            best_obj = obj
            best_beta = beta

    resid = y - X @ best_beta
    s = 1.4826 * np.sqrt(best_obj)
    return ESRes(
        measure="lms",
        estimate=best_obj,
        n=n,
        extra={
            "coefficients": best_beta.tolist(),
            "scale": float(s),
        },
    )


lmses = least_median_squares


def cheatsheet() -> str:
    return "least_median_squares(X, y) -> LMS robust regression."

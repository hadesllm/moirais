# morie.fn -- function file (hadesllm/morie)
"""Least Median of Squares (LMS) robust regression."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def least_median_sq(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_trials: int = 500,
    seed: int | None = None,
) -> DescriptiveResult:
    """LMS regression -- minimises the median of squared residuals.

    Parameters
    ----------
    X : (n, p) array
        Predictor matrix.
    y : (n,) array
        Response.
    n_trials : int
        Random subsamples.
    seed : int, optional

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    k = p + 1

    rng = np.random.default_rng(seed)
    best_med = np.inf
    best_beta = None

    for _ in range(n_trials):
        idx = rng.choice(n, size=min(k, n), replace=False)
        try:
            beta = np.linalg.lstsq(X_int[idx], y[idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        resid2 = (y - X_int @ beta) ** 2
        med = float(np.median(resid2))
        if med < best_med:
            best_med = med
            best_beta = beta

    if best_beta is None:
        best_beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
        best_med = float(np.median((y - X_int @ best_beta) ** 2))

    return DescriptiveResult(
        name="lms",
        value=float(best_med),
        extra={"coefficients": best_beta.tolist(), "n": n, "p": p},
    )


lms = least_median_sq


def cheatsheet() -> str:
    return "least_median_sq({}) -> Least Median of Squares (LMS) robust regression."

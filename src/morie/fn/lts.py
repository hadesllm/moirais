# morie.fn — function file (hadesllm/morie)
"""Least Trimmed Squares (LTS) robust regression."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def least_trimmed_sq(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.5,
    n_trials: int = 500,
    seed: int | None = None,
) -> DescriptiveResult:
    """LTS robust regression estimator.

    Finds the subset of size h = ceil(alpha * n) whose OLS residuals
    have the smallest sum of squares.

    Parameters
    ----------
    X : (n, p) array
        Predictor matrix (intercept NOT added automatically).
    y : (n,) array
        Response.
    alpha : float
        Fraction of observations (0.5 to 1.0).
    n_trials : int
        Random starts.
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
    h = max(int(np.ceil(alpha * n)), p + 1)

    X_int = np.column_stack([np.ones(n), X])
    rng = np.random.default_rng(seed)
    best_ss = np.inf
    best_beta = None

    for _ in range(n_trials):
        idx = rng.choice(n, size=min(p + 2, n), replace=False)
        for _cstep in range(30):
            Xs, ys = X_int[idx], y[idx]
            try:
                beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            resid = (y - X_int @ beta) ** 2
            new_idx = np.argsort(resid)[:h]
            if np.array_equal(np.sort(new_idx), np.sort(idx)):
                break
            idx = new_idx

        ss = np.sort(resid)[:h].sum()
        if ss < best_ss:
            best_ss = ss
            best_beta = beta

    if best_beta is None:
        best_beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
        best_ss = float(np.sort((y - X_int @ best_beta) ** 2)[:h].sum())

    return DescriptiveResult(
        name="lts",
        value=float(best_ss),
        extra={"coefficients": best_beta.tolist(), "alpha": alpha, "h": h, "n": n, "p": p},
    )


lts = least_trimmed_sq


def cheatsheet() -> str:
    return "least_trimmed_sq({}) -> Least Trimmed Squares (LTS) robust regression."

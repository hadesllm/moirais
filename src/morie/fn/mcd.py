# morie.fn — function file (hadesllm/morie)
"""Minimum Covariance Determinant (MCD) estimator."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def min_covariance_det(
    X: np.ndarray,
    *,
    alpha: float = 0.5,
    n_trials: int = 500,
    seed: int | None = None,
) -> DescriptiveResult:
    """Fast MCD estimator for robust location and scatter.

    Simplified C-step algorithm.  Selects the h-subset whose classical
    covariance has the smallest determinant.

    Parameters
    ----------
    X : (n, p) array
        Data matrix.
    alpha : float
        Fraction of observations to keep (0.5 to 1.0).
    n_trials : int
        Number of random initial subsets.
    seed : int, optional
        RNG seed.

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    h = max(int(np.ceil(alpha * n)), p + 1)
    if h > n:
        h = n

    rng = np.random.default_rng(seed)
    best_det = np.inf
    best_loc = np.mean(X, axis=0)
    best_cov = np.cov(X, rowvar=False)

    for _ in range(n_trials):
        idx = rng.choice(n, size=min(p + 1, n), replace=False)
        for _cstep in range(30):
            mu = X[idx].mean(axis=0)
            diff = X - mu
            if len(idx) < 2:
                break
            cov = np.cov(X[idx], rowvar=False)
            if cov.ndim == 0:
                cov = cov.reshape(1, 1)
            try:
                inv_cov = np.linalg.inv(cov + 1e-10 * np.eye(p))
            except np.linalg.LinAlgError:
                break
            dists = np.array([d @ inv_cov @ d for d in diff])
            new_idx = np.argsort(dists)[:h]
            if np.array_equal(np.sort(new_idx), np.sort(idx)):
                break
            idx = new_idx

        mu = X[idx].mean(axis=0)
        cov = np.cov(X[idx], rowvar=False)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        det = np.linalg.det(cov)
        if det < best_det:
            best_det = det
            best_loc = mu
            best_cov = cov

    return DescriptiveResult(
        name="mcd",
        value=float(best_det),
        extra={"location": best_loc.tolist(), "covariance": best_cov.tolist(), "alpha": alpha, "h": h, "n": n, "p": p},
    )


mcd = min_covariance_det


def cheatsheet() -> str:
    return "min_covariance_det({}) -> Minimum Covariance Determinant (MCD) estimator."

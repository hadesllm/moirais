# morie.fn -- function file (hadesllm/morie)
"""Robust covariance (MCD). 'Ironhide reporting.' -- Ironhide"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def robust_covariance_mcd(
    data: np.ndarray,
    *,
    support_fraction: float = 0.75,
    n_iter: int = 100,
    seed: int = 42,
) -> DescriptiveResult:
    """Minimum Covariance Determinant (MCD) robust covariance estimator.

    Implements a simplified C-step MCD algorithm (Rousseeuw & Van Driessen, 1999).
    Finds the subset of ``h = floor(support_fraction * n)`` observations whose
    classical covariance matrix has minimum determinant.

    Parameters
    ----------
    data : ndarray of shape (n, p)
        Input data matrix.
    support_fraction : float
        Fraction of observations to include in the support (0.5 to 1.0).
    n_iter : int
        Number of random starting subsets.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        With ``value`` = robust covariance matrix (p x p) and
        ``extra`` containing location estimate and determinant.
    """
    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if n < p + 1:
        raise ValueError(f"Need n > p: got n={n}, p={p}")
    h = max(p + 1, int(np.floor(support_fraction * n)))
    h = min(h, n)
    rng = np.random.default_rng(seed)

    best_det = np.inf
    best_loc = None
    best_cov = None

    for _ in range(n_iter):
        idx = rng.choice(n, size=h, replace=False)
        for _cstep in range(10):
            subset = X[idx]
            loc = subset.mean(axis=0)
            cov = np.cov(subset, rowvar=False, ddof=1)
            if cov.ndim == 0:
                cov = cov.reshape(1, 1)
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov += np.eye(p) * 1e-8
                cov_inv = np.linalg.inv(cov)
            diff = X - loc
            dists = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
            new_idx = np.argsort(dists)[:h]
            if np.array_equal(np.sort(new_idx), np.sort(idx)):
                break
            idx = new_idx

        det = np.linalg.det(cov)
        if det < best_det:
            best_det = det
            best_loc = loc
            best_cov = cov

    return DescriptiveResult(
        name="MCD_robust_covariance",
        value=best_cov,
        extra={"location": best_loc, "determinant": best_det, "support_fraction": support_fraction, "n": n, "p": p},
    )


ircnh = robust_covariance_mcd


def cheatsheet() -> str:
    return "robust_covariance_mcd({}) -> Robust covariance (MCD). 'Ironhide reporting.' -- Ironhide"

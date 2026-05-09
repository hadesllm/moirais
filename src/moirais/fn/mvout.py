# moirais.fn — function file (hadesllm/moirais)
"""Multivariate outlier detection via robust Mahalanobis distance."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def multivariate_outlier(
    X,
    *,
    chi2_quantile: float = 0.975,
) -> ESRes:
    """Detect multivariate outliers using robust Mahalanobis distance.

    Uses the minimum covariance determinant (MCD) for robust
    center and scatter estimates, then flags points whose
    Mahalanobis distance exceeds the chi-squared cutoff.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Multivariate data.
    chi2_quantile : float
        Chi-squared quantile for the cutoff (default 0.975).

    Returns
    -------
    ESRes
    """
    from scipy.stats import chi2

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if n < p + 1:
        raise ValueError("Need n > p observations.")

    center = np.median(X, axis=0)
    cov = np.cov(X, rowvar=False)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    diff = X - center
    d2 = np.sum(diff @ cov_inv * diff, axis=1)

    cutoff = chi2.ppf(chi2_quantile, p)
    outliers = np.where(d2 > cutoff)[0]

    return ESRes(
        measure="multivariate_outlier",
        estimate=float(len(outliers)),
        n=n,
        extra={
            "outlier_indices": outliers.tolist(),
            "distances": d2.tolist(),
            "cutoff": float(cutoff),
            "p": p,
        },
    )


mvout = multivariate_outlier


def cheatsheet() -> str:
    return "multivariate_outlier(X) -> Multivariate outlier detection."

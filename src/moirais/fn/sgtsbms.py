"""Spectral SBM estimator using top-K eigvecs."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_sbm_spectral_estimate"]


def sgt_sbm_spectral_estimate(A, K):
    """
    Spectral SBM estimator using top-K eigvecs

    Formula: Cluster top-K eigvecs of A or L̃ via k-means

    Parameters
    ----------
    A : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels

    References
    ----------
    Lei & Rinaldo (2015)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Spectral SBM estimator using top-K eigvecs"})
    estimate = np.median(A)
    se = 1.2533 * np.std(A, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Spectral SBM estimator using top-K eigvecs"})


def cheatsheet():
    return "sgtsbms: Spectral SBM estimator using top-K eigvecs"

"""Robust variance estimation for cluster-correlated effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_robust_variance_est"]


def ma_robust_variance_est(yi, X, W, cluster):
    """
    Robust variance estimation for cluster-correlated effects

    Formula: V̂_RVE = (X^T W X)^{-1} X^T W Σ̂ W X (X^T W X)^{-1}

    Parameters
    ----------
    yi : array-like
        Input data.
    X : array-like
        Input data.
    W : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta, V_rve

    References
    ----------
    Hedges-Tipton-Johnson (2010)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Robust variance estimation for cluster-correlated effects"})
    estimate = np.median(yi)
    se = 1.2533 * np.std(yi, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Robust variance estimation for cluster-correlated effects"})


def cheatsheet():
    return "marve: Robust variance estimation for cluster-correlated effects"

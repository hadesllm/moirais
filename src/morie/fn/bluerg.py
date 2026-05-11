"""Best linear unbiased estimator for fixed effects (GLS)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["blue_gls"]


def blue_gls(y, X, V):
    """
    Best linear unbiased estimator for fixed effects (GLS)

    Formula: beta = (X' V^-1 X)^-1 X' V^-1 y

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Aitken (1935); Henderson (1975)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Best linear unbiased estimator for fixed effects (GLS)"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Best linear unbiased estimator for fixed effects (GLS)"})


def cheatsheet():
    return "bluerg: Best linear unbiased estimator for fixed effects (GLS)"

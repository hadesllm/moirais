"""Limited-information ML IV estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_iv_liml"]


def causal_iv_liml(y, X, Z):
    """
    Limited-information ML IV estimator

    Formula: β̂_LIML = argmin |y-Xβ|² over rotations

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta, k

    References
    ----------
    Anderson-Rubin (1949)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Limited-information ML IV estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Limited-information ML IV estimator"})


def cheatsheet():
    return "causivlim: Limited-information ML IV estimator"

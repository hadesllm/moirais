"""Hill tail-index estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hill_estimator"]


def hill_estimator(x, k):
    """
    Hill tail-index estimator

    Formula: ξ̂ = (1/k) sum log(X_(n-i+1) / X_(n-k))

    Parameters
    ----------
    x : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hill (1975)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Hill tail-index estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Hill tail-index estimator"})


def cheatsheet():
    return "hillEst: Hill tail-index estimator"

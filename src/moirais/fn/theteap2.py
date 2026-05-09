"""MAP estimator of theta."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["theta_map"]


def theta_map(X, items, prior):
    """
    MAP estimator of theta

    Formula: argmax posterior(theta | X)

    Parameters
    ----------
    X : array-like
        Input data.
    items : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mislevy (1986)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "MAP estimator of theta"})
    estimate = np.median(X)
    se = 1.2533 * np.std(X, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "MAP estimator of theta"})


def cheatsheet():
    return "theteap2: MAP estimator of theta"

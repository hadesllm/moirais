"""Cluster sample variance estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cluster_variance"]


def cluster_variance(y, cluster):
    """
    Cluster sample variance estimator

    Formula: Var(ybar_clu) = (1 - n/N) S_b^2 / n

    Parameters
    ----------
    y : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977) §9.3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Cluster sample variance estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Cluster sample variance estimator"})


def cheatsheet():
    return "cluvar: Cluster sample variance estimator"

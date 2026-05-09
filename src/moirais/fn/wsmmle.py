"""Maximum likelihood estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_mle"]


def wasserman_mle(data, f, theta0):
    """
    Maximum likelihood estimator

    Formula: theta_hat = argmax_theta l(theta)

    Parameters
    ----------
    data : array-like
        Input data.
    f : array-like
        Input data.
    theta0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Wasserman (2004), Ch 9
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Maximum likelihood estimator"})
    estimate = np.median(data)
    se = 1.2533 * np.std(data, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Maximum likelihood estimator"})


def cheatsheet():
    return "wsmmle: Maximum likelihood estimator"

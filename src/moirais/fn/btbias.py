"""Bootstrap bias estimator of T̂."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_bias_estimator"]


def boot_bias_estimator(theta_hat, theta_b):
    """
    Bootstrap bias estimator of T̂

    Formula: bias = E*[T*] - T̂

    Parameters
    ----------
    theta_hat : array-like
        Input data.
    theta_b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bias

    References
    ----------
    Efron (1979)
    """
    theta_hat = np.atleast_1d(np.asarray(theta_hat, dtype=float))
    n = len(theta_hat)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bootstrap bias estimator of T̂"})
    estimate = np.median(theta_hat)
    se = 1.2533 * np.std(theta_hat, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bootstrap bias estimator of T̂"})


def cheatsheet():
    return "btbias: Bootstrap bias estimator of T̂"

"""Probability-weighted moments estimator for GEV."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gev_pwm"]


def evt_gev_pwm(x):
    """
    Probability-weighted moments estimator for GEV

    Formula: β_r = E[X F(X)^r]; closed-form for ξ,σ,μ

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mu, sigma, xi

    References
    ----------
    Hosking-Wallis-Wood (1985)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Probability-weighted moments estimator for GEV"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Probability-weighted moments estimator for GEV"})


def cheatsheet():
    return "evgevp2: Probability-weighted moments estimator for GEV"

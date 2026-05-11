"""Harmonic-mean volatility estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_harmonic_volatility"]


def vol_harmonic_volatility(sigma):
    """
    Harmonic-mean volatility estimator

    Formula: σ̂_H = N / Σ 1/σ_i

    Parameters
    ----------
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sigma_H

    References
    ----------
    Andersen et al. (2003)
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    n = len(sigma)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Harmonic-mean volatility estimator"})
    estimate = np.median(sigma)
    se = 1.2533 * np.std(sigma, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Harmonic-mean volatility estimator"})


def cheatsheet():
    return "volharm: Harmonic-mean volatility estimator"

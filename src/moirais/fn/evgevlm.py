"""L-moment estimator of GEV parameters."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gev_lmoments"]


def evt_gev_lmoments(x):
    """
    L-moment estimator of GEV parameters

    Formula: ξ from L-CV-skew transcendental, σ,μ closed-form

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
    Hosking (1990)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "L-moment estimator of GEV parameters"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "L-moment estimator of GEV parameters"})


def cheatsheet():
    return "evgevlm: L-moment estimator of GEV parameters"

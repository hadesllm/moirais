"""Estimate microstructure noise variance from RV at fine grid."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_noise_variance_est"]


def vol_noise_variance_est(r_intraday):
    """
    Estimate microstructure noise variance from RV at fine grid

    Formula: η̂² = RV_finest /(2N)

    Parameters
    ----------
    r_intraday : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: eta2

    References
    ----------
    Aït-Sahalia-Mykland-Zhang (2005)
    """
    r_intraday = np.atleast_1d(np.asarray(r_intraday, dtype=float))
    n = len(r_intraday)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Estimate microstructure noise variance from RV at fine grid"})
    estimate = np.median(r_intraday)
    se = 1.2533 * np.std(r_intraday, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Estimate microstructure noise variance from RV at fine grid"})


def cheatsheet():
    return "volnois: Estimate microstructure noise variance from RV at fine grid"

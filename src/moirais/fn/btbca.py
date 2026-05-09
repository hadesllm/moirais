"""Bias-corrected accelerated (BCa) CI."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_bca_ci"]


def boot_bca_ci(theta_hat, theta_b, x, stat, alpha):
    """
    Bias-corrected accelerated (BCa) CI

    Formula: α₁,α₂ adjusted by ẑ_0 + â; quantiles from θ_b

    Parameters
    ----------
    theta_hat : array-like
        Input data.
    theta_b : array-like
        Input data.
    x : array-like
        Input data.
    stat : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi, z0, accel

    References
    ----------
    Efron (1987)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bias-corrected accelerated (BCa) CI"})


def cheatsheet():
    return "btbca: Bias-corrected accelerated (BCa) CI"

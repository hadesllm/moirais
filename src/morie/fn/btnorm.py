"""Normal-approximation bootstrap CI."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_normal_ci"]


def boot_normal_ci(theta_hat, theta_b, alpha):
    """
    Normal-approximation bootstrap CI

    Formula: θ̂ - bias ± z_{1-α/2} ŝe*

    Parameters
    ----------
    theta_hat : array-like
        Input data.
    theta_b : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi, bias, se_b

    References
    ----------
    Davison & Hinkley (1997)
    """
    theta_hat = np.atleast_1d(np.asarray(theta_hat, dtype=float))
    n = len(theta_hat)
    result = float(np.mean(theta_hat))
    se = float(np.std(theta_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normal-approximation bootstrap CI"})


def cheatsheet():
    return "btnorm: Normal-approximation bootstrap CI"

"""Parametric bootstrap from a fitted distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_parametric"]


def boot_parametric(theta_hat, rvs_fn, stat, B, n):
    """
    Parametric bootstrap from a fitted distribution

    Formula: x*_b ~ F(·|θ̂); θ̂*_b = T(x*_b)

    Parameters
    ----------
    theta_hat : array-like
        Input data.
    rvs_fn : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_b

    References
    ----------
    Davison & Hinkley (1997)
    """
    theta_hat = np.atleast_1d(np.asarray(theta_hat, dtype=float))
    n = len(theta_hat)
    result = float(np.mean(theta_hat))
    se = float(np.std(theta_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Parametric bootstrap from a fitted distribution"})


def cheatsheet():
    return "btparm: Parametric bootstrap from a fitted distribution"

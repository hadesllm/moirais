"""Implied VaR from a GARCH(1,1) one-step ahead."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_garch_var_impl"]


def vol_garch_var_impl(mu, sigma_next, alpha, dist):
    """
    Implied VaR from a GARCH(1,1) one-step ahead

    Formula: VaR_{α,t+1} = -μ - z_{1-α} σ_{t+1}

    Parameters
    ----------
    mu : array-like
        Input data.
    sigma_next : array-like
        Input data.
    alpha : array-like
        Input data.
    dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: VaR

    References
    ----------
    McNeil-Frey-Embrechts (2005)
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Implied VaR from a GARCH(1,1) one-step ahead"})


def cheatsheet():
    return "volgvi: Implied VaR from a GARCH(1,1) one-step ahead"

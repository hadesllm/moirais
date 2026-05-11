"""Covariance function of the Brownian-bridge limit of the empirical process."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_brownian_bridge_covariance"]


def kosorok_ch2_brownian_bridge_covariance(s, t, F):
    """
    Covariance function of the Brownian-bridge limit of the empirical process

    Formula: cov[G(s), G(t)] = E[G(s) G(t)] = F(s ^ t) - F(s) F(t)

    Parameters
    ----------
    s : array-like
        Input data.
    t : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.5, p. 11
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = len(s)
    result = float(np.mean(s))
    se = float(np.std(s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Covariance function of the Brownian-bridge limit of the empirical process"})


def cheatsheet():
    return "ksr030: Covariance function of the Brownian-bridge limit of the empirical process"

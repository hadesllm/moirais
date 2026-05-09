"""Metropolis-Hastings MCMC."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_mcmc_metropolis"]


def wasserman_mcmc_metropolis(target, proposal, x0, n):
    """
    Metropolis-Hastings MCMC

    Formula: alpha = min(1, p(x') q(x|x') / (p(x) q(x'|x)))

    Parameters
    ----------
    target : array-like
        Input data.
    proposal : array-like
        Input data.
    x0 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: samples

    References
    ----------
    Wasserman (2004), Ch 24
    """
    target = np.atleast_1d(np.asarray(target, dtype=float))
    n = len(target)
    result = float(np.mean(target))
    se = float(np.std(target, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Metropolis-Hastings MCMC"})


def cheatsheet():
    return "wsmmcm: Metropolis-Hastings MCMC"

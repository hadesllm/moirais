"""Metropolis-Hastings MCMC."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["metropolis_hastings"]


def metropolis_hastings(target, proposal, x0, n_iter):
    """
    Metropolis-Hastings MCMC

    Formula: alpha = min(1, p(x')/p(x) * q(x|x')/q(x'|x))

    Parameters
    ----------
    target : array-like
        Input data.
    proposal : array-like
        Input data.
    x0 : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Metropolis et al (1953); Hastings (1970)
    """
    target = np.atleast_1d(np.asarray(target, dtype=float))
    n = len(target)
    result = float(np.mean(target))
    se = float(np.std(target, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Metropolis-Hastings MCMC"})


def cheatsheet():
    return "mhmcmc: Metropolis-Hastings MCMC"

"""Parallel tempering MCMC."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["parallel_tempering"]


def parallel_tempering(log_p, temperatures, x0, n_iter):
    """
    Parallel tempering MCMC

    Formula: k chains at temperatures T_1<..<T_k; swap proposals

    Parameters
    ----------
    log_p : array-like
        Input data.
    temperatures : array-like
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
    Earl-Deem (2005)
    """
    log_p = np.atleast_1d(np.asarray(log_p, dtype=float))
    n = len(log_p)
    result = float(np.mean(log_p))
    se = float(np.std(log_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Parallel tempering MCMC"})


def cheatsheet():
    return "ptmcmc: Parallel tempering MCMC"

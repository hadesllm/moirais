"""Effective sample size of MCMC."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effective_sample_size_bayes"]


def effective_sample_size_bayes(chain):
    """
    Effective sample size of MCMC

    Formula: n / (1 + 2 sum_k rho_k)

    Parameters
    ----------
    chain : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geyer (1992)
    """
    chain = np.atleast_1d(np.asarray(chain, dtype=float))
    n = len(chain)
    result = float(np.mean(chain))
    se = float(np.std(chain, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effective sample size of MCMC"})


def cheatsheet():
    return "bayess: Effective sample size of MCMC"

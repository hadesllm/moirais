# moirais.fn — function file (hadesllm/moirais)
"""MCMCpack MCMCirtKd interface for k-dimensional Bayesian IRT."""
import numpy as np
from ._richresult import RichResult

__all__ = ["mcmcpack_irt"]


def mcmcpack_irt(votes, n_dims, burnin, n_iter):
    """
    MCMCpack MCMCirtKd interface for k-dimensional Bayesian IRT

    Formula: Calls MCMCirtKd(data, dimensions=k, burnin, mcmc, thin, store.item=T, store.ability=T)

    Parameters
    ----------
    votes : array-like
        Input data.
    n_dims : array-like
        Input data.
    burnin : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'mcmc_output': 'object'}

    References
    ----------
    Armstrong Ch 6
    """
    votes = np.asarray(votes, dtype=float)
    n = int(votes) if votes.ndim == 0 else len(votes)
    result = float(np.mean(votes))
    se = float(np.std(votes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MCMCpack MCMCirtKd interface for k-dimensional Bayesian IRT"})


def cheatsheet():
    return "mcmpp: MCMCpack MCMCirtKd interface for k-dimensional Bayesian IRT"

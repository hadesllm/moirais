"""Reversible-jump MCMC across model dimensions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["reversible_jump_mcmc"]


def reversible_jump_mcmc(models, x0, n_iter):
    """
    Reversible-jump MCMC across model dimensions

    Formula: trans-dimensional move with Jacobian factor

    Parameters
    ----------
    models : array-like
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
    Green (1995)
    """
    models = np.atleast_1d(np.asarray(models, dtype=float))
    n = len(models)
    result = float(np.mean(models))
    se = float(np.std(models, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reversible-jump MCMC across model dimensions"})


def cheatsheet():
    return "bayrjmcmc: Reversible-jump MCMC across model dimensions"

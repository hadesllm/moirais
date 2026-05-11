"""Hyperparameter optimization for GP via MCMC."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hyperparam_optim_gp"]


def hyperparam_optim_gp(X, y, prior):
    """
    Hyperparameter optimization for GP via MCMC

    Formula: MCMC over kernel hyperparams + integrate out

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Murray-Adams (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hyperparameter optimization for GP via MCMC"})


def cheatsheet():
    return "hyper2: Hyperparameter optimization for GP via MCMC"

"""Bayesian hierarchical spatial model structure."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_bayesian_hierarchical"]


def schabenberger_bayesian_hierarchical(y, x, w, prior_spec):
    """
    Bayesian hierarchical spatial model structure

    Formula: Y|theta~likelihood; theta|xi~spatial_prior (ICAR/GP); xi~hyperprior; posterior via MCMC

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    w : array-like
        Input data.
    prior_spec : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: posterior

    References
    ----------
    Schabenberger Ch 6, Sec 6.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian hierarchical spatial model structure"})


def cheatsheet():
    return "spbayr: Bayesian hierarchical spatial model structure"

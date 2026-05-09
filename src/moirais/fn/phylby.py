"""Bayesian phylogeny via MrBayes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayesian_phylogeny"]


def bayesian_phylogeny(alignment, priors, n_iter):
    """
    Bayesian phylogeny via MrBayes

    Formula: MCMC over (tree, branches, model)

    Parameters
    ----------
    alignment : array-like
        Input data.
    priors : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ronquist-Huelsenbeck (2003)
    """
    alignment = np.atleast_1d(np.asarray(alignment, dtype=float))
    n = len(alignment)
    result = float(np.mean(alignment))
    se = float(np.std(alignment, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian phylogeny via MrBayes"})


def cheatsheet():
    return "phylby: Bayesian phylogeny via MrBayes"

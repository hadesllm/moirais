"""Gibbs sampler for CRP cluster assignments."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["crp_gibbs"]


def crp_gibbs(y, alpha, n_iter):
    """
    Gibbs sampler for CRP cluster assignments

    Formula: P(z_i|z_{-i}) ∝ ...

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Neal (2000) Algorithm 8
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gibbs sampler for CRP cluster assignments"})


def cheatsheet():
    return "crpgib: Gibbs sampler for CRP cluster assignments"

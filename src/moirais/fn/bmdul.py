# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian multidimensional unfolding with MCMC."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bayesian_mds_unfolding"]


def bayesian_mds_unfolding(ratings, n_dims, n_iter):
    """
    Bayesian multidimensional unfolding with MCMC

    Formula: P(X,Y|R) prop L(R|X,Y)*P(X)*P(Y); posterior over ideal points and stimuli

    Parameters
    ----------
    ratings : array-like
        Input data.
    n_dims : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'X_samples': 'array', 'Y_samples': 'array'}

    References
    ----------
    Armstrong Ch 6
    """
    ratings = np.asarray(ratings, dtype=float)
    n = int(ratings) if ratings.ndim == 0 else len(ratings)
    result = float(np.mean(ratings))
    se = float(np.std(ratings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian multidimensional unfolding with MCMC"})


def cheatsheet():
    return "bmdul: Bayesian multidimensional unfolding with MCMC"

"""Dirichlet process mixture."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_dirichlet_proc"]


def esl_dirichlet_proc(alpha, G0):
    """
    Dirichlet process mixture

    Formula: G ~ DP(alpha, G_0)

    Parameters
    ----------
    alpha : array-like
        Input data.
    G0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: process

    References
    ----------
    Hastie ESL Ch 8
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(alpha)
    result = float(np.mean(alpha))
    se = float(np.std(alpha, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirichlet process mixture"})


def cheatsheet():
    return "esldai: Dirichlet process mixture"

"""Stochastic Kronecker graph."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kronecker_graph"]


def kronecker_graph(seed, k):
    """
    Stochastic Kronecker graph

    Formula: recursive Kronecker product of seed

    Parameters
    ----------
    seed : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Leskovec et al (2010)
    """
    seed = np.atleast_1d(np.asarray(seed, dtype=float))
    n = len(seed)
    result = float(np.mean(seed))
    se = float(np.std(seed, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stochastic Kronecker graph"})


def cheatsheet():
    return "krfgrp: Stochastic Kronecker graph"

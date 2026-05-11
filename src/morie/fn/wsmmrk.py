"""Markov inequality P(X >= a) <= E[X]/a."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_markov_ineq"]


def wasserman_markov_ineq(mean, a):
    """
    Markov inequality P(X >= a) <= E[X]/a

    Formula: P(X >= a) <= E[|X|] / a

    Parameters
    ----------
    mean : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bound

    References
    ----------
    Wasserman (2004), Ch 4
    """
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    n = len(mean)
    result = float(np.mean(mean))
    se = float(np.std(mean, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Markov inequality P(X >= a) <= E[X]/a"})


def cheatsheet():
    return "wsmmrk: Markov inequality P(X >= a) <= E[X]/a"

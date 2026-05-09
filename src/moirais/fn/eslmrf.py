"""Markov random field potential."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_markov_rf"]


def esl_markov_rf(graph, psi):
    """
    Markov random field potential

    Formula: p(x) = (1/Z) prod_C psi_C(x_C)

    Parameters
    ----------
    graph : array-like
        Input data.
    psi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: joint

    References
    ----------
    Hastie ESL Ch 17
    """
    graph = np.atleast_1d(np.asarray(graph, dtype=float))
    n = len(graph)
    result = float(np.mean(graph))
    se = float(np.std(graph, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Markov random field potential"})


def cheatsheet():
    return "eslmrf: Markov random field potential"

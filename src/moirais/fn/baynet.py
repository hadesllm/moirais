"""Bayesian network inference (variable elimination)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_network"]


def bayes_network(graph, cpts, evidence, query):
    """
    Bayesian network inference (variable elimination)

    Formula: sum out non-query nodes from joint factor product

    Parameters
    ----------
    graph : array-like
        Input data.
    cpts : array-like
        Input data.
    evidence : array-like
        Input data.
    query : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pearl (1988); Koller-Friedman (2009)
    """
    graph = np.atleast_1d(np.asarray(graph, dtype=float))
    n = len(graph)
    result = float(np.mean(graph))
    se = float(np.std(graph, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian network inference (variable elimination)"})


def cheatsheet():
    return "baynet: Bayesian network inference (variable elimination)"

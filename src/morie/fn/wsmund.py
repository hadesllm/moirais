"""Undirected graphical model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_undirected_graph"]


def wasserman_undirected_graph(graph, psi):
    """
    Undirected graphical model

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
    Wasserman (2004), Ch 17
    """
    graph = np.atleast_1d(np.asarray(graph, dtype=float))
    n = len(graph)
    result = float(np.mean(graph))
    se = float(np.std(graph, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Undirected graphical model"})


def cheatsheet():
    return "wsmund: Undirected graphical model"

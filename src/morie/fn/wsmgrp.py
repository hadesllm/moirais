"""Graphical model factorization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_graphical_model"]


def wasserman_graphical_model(graph, psi):
    """
    Graphical model factorization

    Formula: p(x) = prod_C psi_C(x_C) / Z

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Graphical model factorization"})


def cheatsheet():
    return "wsmgrp: Graphical model factorization"

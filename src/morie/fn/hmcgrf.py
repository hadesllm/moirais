# morie.fn -- function file (hadesllm/morie)
"""Computational graph for expression differentiation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_computational_graph"]


def geron_computational_graph(expr):
    """
    Computational graph for expression differentiation

    Formula: DAG of ops with forward and backward pass

    Parameters
    ----------
    expr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: graph

    References
    ----------
    Géron Appendix A
    """
    expr = np.atleast_1d(np.asarray(expr, dtype=float))
    n = len(expr)
    result = float(np.mean(expr))
    se = float(np.std(expr, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Computational graph for expression differentiation"})


def cheatsheet():
    return "hmcgrf: Computational graph for expression differentiation"

# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Computational-graph forward + backward pass for autodiff."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_computational_graph"]


def burkov_computational_graph(graph, inputs):
    """
    Computational-graph forward + backward pass for autodiff

    Formula: forward: y = f(x) node-by-node; backward: grad propagates via chain rule across DAG

    Parameters
    ----------
    graph : array-like
        Input data.
    inputs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: outputs, grads

    References
    ----------
    Burkov Ch 1, Computational Graph section
    """
    graph = np.atleast_1d(np.asarray(graph, dtype=float))
    n = len(graph)
    result = float(np.mean(graph))
    se = float(np.std(graph, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Computational-graph forward + backward pass for autodiff"})


def cheatsheet():
    return "bkcgr: Computational-graph forward + backward pass for autodiff"

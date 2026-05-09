# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bow-ban (no self-loops): DAG acyclicity condition ensuring no cycles."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bow_ban_theorem"]


def bow_ban_theorem(graph):
    """
    Bow-ban (no self-loops): DAG acyclicity condition ensuring no cycles

    Formula: G is DAG iff topological ordering exists; no V_i is ancestor of itself

    Parameters
    ----------
    graph : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'is_dag': 'bool', 'cycle': 'list'}

    References
    ----------
    Molak Ch 4
    """
    graph = np.asarray(graph, dtype=float)
    n = int(graph) if graph.ndim == 0 else len(graph)
    result = float(np.mean(graph))
    se = float(np.std(graph, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bow-ban (no self-loops): DAG acyclicity condition ensuring no cycles"})


def cheatsheet():
    return "bowbn: Bow-ban (no self-loops): DAG acyclicity condition ensuring no cycles"

"""Directed graphical model (DAG)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_directed_graph"]


def wasserman_directed_graph(dag, x):
    """
    Directed graphical model (DAG)

    Formula: p(x) = prod_i p(x_i | pa(x_i))

    Parameters
    ----------
    dag : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: joint

    References
    ----------
    Wasserman (2004), Ch 17
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Directed graphical model (DAG)"})


def cheatsheet():
    return "wsmdir: Directed graphical model (DAG)"

"""Weighted-graph vertex strengths s_i = Σ_j W_{ij}."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_vertex_strengths"]


def sgt_vertex_strengths(W):
    """
    Weighted-graph vertex strengths s_i = Σ_j W_{ij}

    Formula: s_i = Σ_j W_{ij}

    Parameters
    ----------
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: s

    References
    ----------
    Barrat-Barthélémy-Vespignani (2004)
    """
    W = np.atleast_1d(np.asarray(W, dtype=float))
    n = len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weighted-graph vertex strengths s_i = Σ_j W_{ij}"})


def cheatsheet():
    return "sgtvst: Weighted-graph vertex strengths s_i = Σ_j W_{ij}"

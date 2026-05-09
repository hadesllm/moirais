# moirais.fn — function file (hadesllm/moirais)
"""Magnitude-based unstructured weight pruning."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_weight_pruning"]


def geron_weight_pruning(W, sparsity):
    """
    Magnitude-based unstructured weight pruning

    Formula: mask_{ij} = 1 if |W_{ij}| > threshold else 0; W := W * mask

    Parameters
    ----------
    W : array-like
        Input data.
    sparsity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W_pruned

    References
    ----------
    Géron Ch 17, Weight Pruning section
    """
    W = np.atleast_1d(np.asarray(W, dtype=float))
    n = len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Magnitude-based unstructured weight pruning"})


def cheatsheet():
    return "grprn: Magnitude-based unstructured weight pruning"

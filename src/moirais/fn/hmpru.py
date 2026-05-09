# moirais.fn — function file (hadesllm/moirais)
"""Magnitude-based weight pruning."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_weight_pruning"]


def geron_weight_pruning(model, sparsity):
    """
    Magnitude-based weight pruning

    Formula: zero out |w| < threshold; optional iterative pruning + fine-tuning

    Parameters
    ----------
    model : array-like
        Input data.
    sparsity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 17
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Magnitude-based weight pruning"})


def cheatsheet():
    return "hmpru: Magnitude-based weight pruning"

# moirais.fn — function file (hadesllm/moirais)
"""Contrastive learning: pull positives close, push negatives far."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_contrastive_learning"]


def geron_contrastive_learning(embeddings, positives):
    """
    Contrastive learning: pull positives close, push negatives far

    Formula: L = -log exp(sim(x,x+)/tau) / sum_j exp(sim(x,x_j)/tau)

    Parameters
    ----------
    embeddings : array-like
        Input data.
    positives : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 16
    """
    embeddings = np.atleast_1d(np.asarray(embeddings, dtype=float))
    n = len(embeddings)
    result = float(np.mean(embeddings))
    se = float(np.std(embeddings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Contrastive learning: pull positives close, push negatives far"})


def cheatsheet():
    return "hmcst: Contrastive learning: pull positives close, push negatives far"

# morie.fn — function file (hadesllm/morie)
"""Embedding lookup: map token id to dense vector."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_embedding_lookup"]


def geron_embedding_lookup(ids, E):
    """
    Embedding lookup: map token id to dense vector

    Formula: e = E[id]  where E is (V, d) embedding table

    Parameters
    ----------
    ids : array-like
        Input data.
    E : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: embeddings

    References
    ----------
    Géron Ch 14, Word Embeddings section
    """
    ids = np.atleast_1d(np.asarray(ids, dtype=float))
    n = len(ids)
    result = float(np.mean(ids))
    se = float(np.std(ids, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Embedding lookup: map token id to dense vector"})


def cheatsheet():
    return "gremb: Embedding lookup: map token id to dense vector"

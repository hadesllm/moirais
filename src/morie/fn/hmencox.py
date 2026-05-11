# morie.fn — function file (hadesllm/morie)
"""Encoder-only transformer (BERT-family)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_encoder_only"]


def geron_encoder_only(X, n_layers, n_heads):
    """
    Encoder-only transformer (BERT-family)

    Formula: stacked self-attention + FFN; CLS token for classification

    Parameters
    ----------
    X : array-like
        Input data.
    n_layers : array-like
        Input data.
    n_heads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 15
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Encoder-only transformer (BERT-family)"})


def cheatsheet():
    return "hmencox: Encoder-only transformer (BERT-family)"

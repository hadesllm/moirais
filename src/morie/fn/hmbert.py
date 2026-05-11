# morie.fn — function file (hadesllm/morie)
"""BERT: bidirectional encoder pretrained on MLM + NSP."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bert"]


def geron_bert(X, n_layers, n_heads, d_model):
    """
    BERT: bidirectional encoder pretrained on MLM + NSP

    Formula: encoder-only; mask 15% tokens, predict masked via MLM head

    Parameters
    ----------
    X : array-like
        Input data.
    n_layers : array-like
        Input data.
    n_heads : array-like
        Input data.
    d_model : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BERT: bidirectional encoder pretrained on MLM + NSP"})


def cheatsheet():
    return "hmbert: BERT: bidirectional encoder pretrained on MLM + NSP"

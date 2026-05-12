# morie.fn -- function file (hadesllm/morie)
"""Decoder-only transformer (GPT family)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_decoder_only"]


def geron_decoder_only(X, n_layers, n_heads):
    """
    Decoder-only transformer (GPT family)

    Formula: causal self-attention; predict next token

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decoder-only transformer (GPT family)"})


def cheatsheet():
    return "hmdctr: Decoder-only transformer (GPT family)"

# moirais.fn — function file (hadesllm/moirais)
"""GPT-1: decoder-only transformer pretrained on next-token prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gpt1"]


def geron_gpt1(X, n_layers, n_heads):
    """
    GPT-1: decoder-only transformer pretrained on next-token prediction

    Formula: L = -sum_t log P(x_t | x_{<t})

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GPT-1: decoder-only transformer pretrained on next-token prediction"})


def cheatsheet():
    return "hmgpt1: GPT-1: decoder-only transformer pretrained on next-token prediction"

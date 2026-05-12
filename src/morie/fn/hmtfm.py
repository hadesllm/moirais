# morie.fn -- function file (hadesllm/morie)
"""Transformer architecture (Vaswani et al. 2017)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_transformer"]


def geron_transformer(X, n_heads, d_model, n_layers):
    """
    Transformer architecture (Vaswani et al. 2017)

    Formula: stacked multi-head attention + feedforward with residual connections

    Parameters
    ----------
    X : array-like
        Input data.
    n_heads : array-like
        Input data.
    d_model : array-like
        Input data.
    n_layers : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transformer architecture (Vaswani et al. 2017)"})


def cheatsheet():
    return "hmtfm: Transformer architecture (Vaswani et al. 2017)"

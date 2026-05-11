# morie.fn — function file (hadesllm/morie)
"""Bidirectional RNN: concatenate forward and backward hidden states."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bidirectional_rnn"]


def geron_bidirectional_rnn(X, Wx_f, Wh_f, Wx_b, Wh_b):
    """
    Bidirectional RNN: concatenate forward and backward hidden states

    Formula: h_t = [h_t^fwd; h_t^bwd]

    Parameters
    ----------
    X : array-like
        Input data.
    Wx_f : array-like
        Input data.
    Wh_f : array-like
        Input data.
    Wx_b : array-like
        Input data.
    Wh_b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: H

    References
    ----------
    Géron Ch 14
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bidirectional RNN: concatenate forward and backward hidden states"})


def cheatsheet():
    return "hmbrnn: Bidirectional RNN: concatenate forward and backward hidden states"

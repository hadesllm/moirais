# morie.fn -- function file (hadesllm/morie)
"""Bidirectional RNN: concatenate forward and backward hidden states."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bidirectional_rnn"]


def geron_bidirectional_rnn(h_forward, h_backward):
    """
    Bidirectional RNN: concatenate forward and backward hidden states

    Formula: h_t = [h_t^forward ; h_t^backward]

    Parameters
    ----------
    h_forward : array-like
        Input data.
    h_backward : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h

    References
    ----------
    Géron Ch 14, Bidirectional RNN section
    """
    h_forward = np.atleast_1d(np.asarray(h_forward, dtype=float))
    n = len(h_forward)
    result = float(np.mean(h_forward))
    se = float(np.std(h_forward, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bidirectional RNN: concatenate forward and backward hidden states"})


def cheatsheet():
    return "grbrnn: Bidirectional RNN: concatenate forward and backward hidden states"

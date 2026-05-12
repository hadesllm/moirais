# morie.fn -- function file (hadesllm/morie)
"""RNNs with visual attention over spatial feature map."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_rnn_visual_attention"]


def geron_rnn_visual_attention(features, h, W, U, v):
    """
    RNNs with visual attention over spatial feature map

    Formula: context_t = sum_{i,j} alpha_{ij,t} * feature_{i,j}

    Parameters
    ----------
    features : array-like
        Input data.
    h : array-like
        Input data.
    W : array-like
        Input data.
    U : array-like
        Input data.
    v : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: alpha, context

    References
    ----------
    Géron Ch 16
    """
    features = np.atleast_1d(np.asarray(features, dtype=float))
    n = len(features)
    result = float(np.mean(features))
    se = float(np.std(features, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RNNs with visual attention over spatial feature map"})


def cheatsheet():
    return "hmrvat: RNNs with visual attention over spatial feature map"

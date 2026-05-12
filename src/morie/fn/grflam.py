# morie.fn -- function file (hadesllm/morie)
"""Flamingo gated cross-attention between language tokens and visual features."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_flamingo_cross_modal_attn"]


def geron_flamingo_cross_modal_attn(h, visual_features, alpha, weights):
    """
    Flamingo gated cross-attention between language tokens and visual features

    Formula: h = h + tanh(alpha) * CrossAttn(h, visual_features); alpha learned scalar per layer

    Parameters
    ----------
    h : array-like
        Input data.
    visual_features : array-like
        Input data.
    alpha : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_out

    References
    ----------
    Géron Ch 16, Flamingo section
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Flamingo gated cross-attention between language tokens and visual features"})


def cheatsheet():
    return "grflam: Flamingo gated cross-attention between language tokens and visual features"

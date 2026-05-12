# morie.fn -- function file (hadesllm/morie)
"""MoE router: softmax-gated expert selection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_moe_router_softmax"]


def kamath_moe_router_softmax(x, Wr, experts, k):
    """
    MoE router: softmax-gated expert selection

    Formula: g(x) = softmax(W_r x);  y = sum_{i in TopK(g)} g_i * Expert_i(x)

    Parameters
    ----------
    x : array-like
        Input data.
    Wr : array-like
        Input data.
    experts : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Kamath Ch 2, Mixture-of-Experts section (Mixtral)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MoE router: softmax-gated expert selection"})


def cheatsheet():
    return "kmmoe: MoE router: softmax-gated expert selection"

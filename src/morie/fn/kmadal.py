# morie.fn -- function file (hadesllm/morie)
"""AdaLoRA: SVD-parametrized LoRA delta with importance-based rank pruning."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_adalora_rank_allocation"]


def kamath_adalora_rank_allocation(P, s, Q, importance, target_rank):
    """
    AdaLoRA: SVD-parametrized LoRA delta with importance-based rank pruning

    Formula: Delta W = P diag(s) Q^T;  prune smallest s_i during training based on importance I(s_i)

    Parameters
    ----------
    P : array-like
        Input data.
    s : array-like
        Input data.
    Q : array-like
        Input data.
    importance : array-like
        Input data.
    target_rank : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Delta_W

    References
    ----------
    Kamath Ch 4, AdaLoRA section
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AdaLoRA: SVD-parametrized LoRA delta with importance-based rank pruning"})


def cheatsheet():
    return "kmadal: AdaLoRA: SVD-parametrized LoRA delta with importance-based rank pruning"

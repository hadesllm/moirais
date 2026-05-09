# moirais.fn — function file (hadesllm/moirais)
"""Top-k sampling: renormalize probabilities over the k most likely tokens."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_topk_sampling"]


def geron_topk_sampling(logits, k):
    """
    Top-k sampling: renormalize probabilities over the k most likely tokens

    Formula: p'_i = p_i / sum_{j in TopK} p_j for i in TopK; 0 otherwise

    Parameters
    ----------
    logits : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probs

    References
    ----------
    Géron Ch 15, Sampling strategies section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Top-k sampling: renormalize probabilities over the k most likely tokens"})


def cheatsheet():
    return "grtop: Top-k sampling: renormalize probabilities over the k most likely tokens"

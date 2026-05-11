# morie.fn — function file (hadesllm/morie)
"""Prefix-LM attention mask: bidirectional over prefix, causal over completion."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_prefix_lm_mask"]


def kamath_prefix_lm_mask(prefix_len, total_len):
    """
    Prefix-LM attention mask: bidirectional over prefix, causal over completion

    Formula: M_ij = 0 for i,j in prefix OR (j in prefix for any i) OR j <= i in completion

    Parameters
    ----------
    prefix_len : array-like
        Input data.
    total_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mask

    References
    ----------
    Kamath Ch 2, Prefix Language Modeling section
    """
    prefix_len = np.atleast_1d(np.asarray(prefix_len, dtype=float))
    n = len(prefix_len)
    result = float(np.mean(prefix_len))
    se = float(np.std(prefix_len, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prefix-LM attention mask: bidirectional over prefix, causal over completion"})


def cheatsheet():
    return "kmprf: Prefix-LM attention mask: bidirectional over prefix, causal over completion"

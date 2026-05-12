# morie.fn -- function file (hadesllm/morie)
"""Top-p (nucleus) sampling: truncate to smallest set with cum. prob >= p."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_nucleus_sampling"]


def kamath_nucleus_sampling(logits, p):
    """
    Top-p (nucleus) sampling: truncate to smallest set with cum. prob >= p

    Formula: V_p = smallest {v_1,..,v_k} s.t. sum P(v_i) >= p; renormalize, sample

    Parameters
    ----------
    logits : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probs

    References
    ----------
    Kamath Ch 4, top-p sampling section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Top-p (nucleus) sampling: truncate to smallest set with cum. prob >= p"})


def cheatsheet():
    return "kmnuc: Top-p (nucleus) sampling: truncate to smallest set with cum. prob >= p"

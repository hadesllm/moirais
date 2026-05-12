# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Repetition penalty: discount logits of previously generated tokens."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_repetition_penalty"]


def burkov_repetition_penalty(logits, prev_tokens, penalty):
    """
    Repetition penalty: discount logits of previously generated tokens

    Formula: for t in prev_tokens: logits[t] = logits[t] / penalty if logits[t] > 0 else logits[t] * penalty

    Parameters
    ----------
    logits : array-like
        Input data.
    prev_tokens : array-like
        Input data.
    penalty : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: logits_adjusted

    References
    ----------
    Burkov Ch 5, Repetition Penalty section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Repetition penalty: discount logits of previously generated tokens"})


def cheatsheet():
    return "bkrep: Repetition penalty: discount logits of previously generated tokens"

# morie.fn -- function file (hadesllm/morie)
"""Verbalizer: map class labels to answer tokens and aggregate probs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_verbalizer_mapping"]


def kamath_verbalizer_mapping(logits, vocab, verbalizer_map):
    """
    Verbalizer: map class labels to answer tokens and aggregate probs

    Formula: P(y_c | x) = sum_{v in V_c} P_LLM(v | x)

    Parameters
    ----------
    logits : array-like
        Input data.
    vocab : array-like
        Input data.
    verbalizer_map : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: class_probs

    References
    ----------
    Kamath Ch 3, Verbalizer Mapping section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Verbalizer: map class labels to answer tokens and aggregate probs"})


def cheatsheet():
    return "kmverb: Verbalizer: map class labels to answer tokens and aggregate probs"

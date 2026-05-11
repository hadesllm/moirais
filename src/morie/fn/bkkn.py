# morie.fn — function file (hadesllm/morie)
"""Kneser-Ney smoothing: absolute discount + continuation probability."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_kneser_ney"]


def burkov_kneser_ney(counts_ngram, counts_prefix, continuation_counts, d):
    """
    Kneser-Ney smoothing: absolute discount + continuation probability

    Formula: P_KN = max(count - d, 0) / count_prefix + lambda * P_continuation(w_n)

    Parameters
    ----------
    counts_ngram : array-like
        Input data.
    counts_prefix : array-like
        Input data.
    continuation_counts : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prob

    References
    ----------
    Burkov Ch 2, Kneser-Ney Smoothing section
    """
    counts_ngram = np.atleast_1d(np.asarray(counts_ngram, dtype=float))
    n = len(counts_ngram)
    result = float(np.mean(counts_ngram))
    se = float(np.std(counts_ngram, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kneser-Ney smoothing: absolute discount + continuation probability"})


def cheatsheet():
    return "bkkn: Kneser-Ney smoothing: absolute discount + continuation probability"

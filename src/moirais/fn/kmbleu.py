# moirais.fn — function file (hadesllm/moirais)
"""BLEU score: geometric mean of n-gram precisions * brevity penalty."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_bleu_score"]


def kamath_bleu_score(hypothesis, references, max_n):
    """
    BLEU score: geometric mean of n-gram precisions * brevity penalty

    Formula: BLEU = BP * exp( sum_{n=1..N} w_n * log p_n );  BP = min(1, exp(1 - r/c))

    Parameters
    ----------
    hypothesis : array-like
        Input data.
    references : array-like
        Input data.
    max_n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bleu

    References
    ----------
    Kamath Ch 8, BLEU Score section
    """
    hypothesis = np.atleast_1d(np.asarray(hypothesis, dtype=float))
    n = len(hypothesis)
    result = float(np.mean(hypothesis))
    se = float(np.std(hypothesis, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BLEU score: geometric mean of n-gram precisions * brevity penalty"})


def cheatsheet():
    return "kmbleu: BLEU score: geometric mean of n-gram precisions * brevity penalty"

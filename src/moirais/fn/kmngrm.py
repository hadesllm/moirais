# moirais.fn — function file (hadesllm/moirais)
"""N-gram language model conditional probability (MLE)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_ngram_language_model"]


def kamath_ngram_language_model(counts_ngram, counts_prefix):
    """
    N-gram language model conditional probability (MLE)

    Formula: P(w_t | w_{t-n+1..t-1}) = count(w_{t-n+1..t}) / count(w_{t-n+1..t-1})

    Parameters
    ----------
    counts_ngram : array-like
        Input data.
    counts_prefix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prob

    References
    ----------
    Kamath Ch 1, N-gram Language Models section
    """
    counts_ngram = np.atleast_1d(np.asarray(counts_ngram, dtype=float))
    n = len(counts_ngram)
    result = float(np.mean(counts_ngram))
    se = float(np.std(counts_ngram, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "N-gram language model conditional probability (MLE)"})


def cheatsheet():
    return "kmngrm: N-gram language model conditional probability (MLE)"

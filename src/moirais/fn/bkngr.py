# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""N-gram maximum-likelihood probability (uninsmoothed)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_ngram_mle"]


def burkov_ngram_mle(counts_ngram, counts_prefix):
    """
    N-gram maximum-likelihood probability (uninsmoothed)

    Formula: P_MLE(w_n | w_{1..n-1}) = count(w_{1..n}) / count(w_{1..n-1})

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
    Burkov Ch 2, Maximum Likelihood Estimation for N-grams section
    """
    counts_ngram = np.atleast_1d(np.asarray(counts_ngram, dtype=float))
    n = len(counts_ngram)
    result = float(np.mean(counts_ngram))
    se = float(np.std(counts_ngram, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "N-gram maximum-likelihood probability (uninsmoothed)"})


def cheatsheet():
    return "bkngr: N-gram maximum-likelihood probability (uninsmoothed)"

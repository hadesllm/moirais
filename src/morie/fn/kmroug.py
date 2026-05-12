# morie.fn -- function file (hadesllm/morie)
"""ROUGE-N recall: n-gram recall of hypothesis against reference."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_rouge_n"]


def kamath_rouge_n(hypothesis, reference, n):
    """
    ROUGE-N recall: n-gram recall of hypothesis against reference

    Formula: ROUGE-N = sum_{g in ngrams(ref)} count_match(g) / sum_{g in ngrams(ref)} count(g)

    Parameters
    ----------
    hypothesis : array-like
        Input data.
    reference : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rouge_n

    References
    ----------
    Kamath Ch 8, ROUGE section
    """
    hypothesis = np.atleast_1d(np.asarray(hypothesis, dtype=float))
    n = len(hypothesis)
    result = float(np.mean(hypothesis))
    se = float(np.std(hypothesis, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ROUGE-N recall: n-gram recall of hypothesis against reference"})


def cheatsheet():
    return "kmroug: ROUGE-N recall: n-gram recall of hypothesis against reference"

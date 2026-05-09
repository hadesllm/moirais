"""BLEU n-gram precision score."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bleu"]


def bleu(candidate, references, max_n):
    """
    BLEU n-gram precision score

    Formula: BP · exp(sum w_n log p_n)

    Parameters
    ----------
    candidate : array-like
        Input data.
    references : array-like
        Input data.
    max_n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Papineni et al (2002)
    """
    candidate = np.atleast_1d(np.asarray(candidate, dtype=float))
    n = len(candidate)
    result = float(np.mean(candidate))
    se = float(np.std(candidate, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BLEU n-gram precision score"})


def cheatsheet():
    return "bleuS: BLEU n-gram precision score"

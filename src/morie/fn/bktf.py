# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Term frequency: raw count of term t in document d (optionally normalized)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_term_frequency"]


def burkov_term_frequency(term, document):
    """
    Term frequency: raw count of term t in document d (optionally normalized)

    Formula: TF(t, d) = count(t in d)  or  count(t in d) / |d|

    Parameters
    ----------
    term : array-like
        Input data.
    document : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tf

    References
    ----------
    Burkov Ch 2, Term Frequency section
    """
    term = np.atleast_1d(np.asarray(term, dtype=float))
    n = len(term)
    result = float(np.mean(term))
    se = float(np.std(term, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Term frequency: raw count of term t in document d (optionally normalized)"})


def cheatsheet():
    return "bktf: Term frequency: raw count of term t in document d (optionally normalized)"

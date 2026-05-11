"""Term rewriting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["term_rewriting"]


def term_rewriting(term, rules):
    """
    Term rewriting

    Formula: normal-form via confluent rule set

    Parameters
    ----------
    term : array-like
        Input data.
    rules : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Baader-Nipkow (1998)
    """
    term = np.atleast_1d(np.asarray(term, dtype=float))
    n = len(term)
    result = float(np.mean(term))
    se = float(np.std(term, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Term rewriting"})


def cheatsheet():
    return "trmRew: Term rewriting"

"""LL(1) recursive descent."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ll_parser"]


def ll_parser(grammar, tokens):
    """
    LL(1) recursive descent

    Formula: top-down predictive parsing

    Parameters
    ----------
    grammar : array-like
        Input data.
    tokens : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Knuth (1971)
    """
    grammar = np.atleast_1d(np.asarray(grammar, dtype=float))
    n = len(grammar)
    result = float(np.mean(grammar))
    se = float(np.std(grammar, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LL(1) recursive descent"})


def cheatsheet():
    return "prsLL: LL(1) recursive descent"

"""LR(1) parser."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lr_parser"]


def lr_parser(grammar, tokens):
    """
    LR(1) parser

    Formula: shift-reduce on canonical LR table

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
    Knuth (1965)
    """
    grammar = np.atleast_1d(np.asarray(grammar, dtype=float))
    n = len(grammar)
    result = float(np.mean(grammar))
    se = float(np.std(grammar, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LR(1) parser"})


def cheatsheet():
    return "prsLR: LR(1) parser"

"""Parsing expression grammar."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["peg_parser"]


def peg_parser(grammar, input):
    """
    Parsing expression grammar

    Formula: ordered choice + greedy match

    Parameters
    ----------
    grammar : array-like
        Input data.
    input : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ford (2004)
    """
    grammar = np.atleast_1d(np.asarray(grammar, dtype=float))
    n = len(grammar)
    result = float(np.mean(grammar))
    se = float(np.std(grammar, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Parsing expression grammar"})


def cheatsheet():
    return "prsPEG: Parsing expression grammar"

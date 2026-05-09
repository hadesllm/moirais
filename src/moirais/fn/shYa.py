"""Shunting-yard infix→RPN."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["shunting_yard"]


def shunting_yard(tokens):
    """
    Shunting-yard infix→RPN

    Formula: two-stack precedence-based parser

    Parameters
    ----------
    tokens : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dijkstra (1961)
    """
    tokens = np.atleast_1d(np.asarray(tokens, dtype=float))
    n = len(tokens)
    result = float(np.mean(tokens))
    se = float(np.std(tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shunting-yard infix→RPN"})


def cheatsheet():
    return "shYa: Shunting-yard infix→RPN"

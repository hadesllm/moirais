"""Reverse-Polish (postfix) evaluation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prefix_evaluation"]


def prefix_evaluation(expr):
    """
    Reverse-Polish (postfix) evaluation

    Formula: stack-machine numeric eval

    Parameters
    ----------
    expr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Łukasiewicz (1924)
    """
    expr = np.atleast_1d(np.asarray(expr, dtype=float))
    n = len(expr)
    result = float(np.mean(expr))
    se = float(np.std(expr, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reverse-Polish (postfix) evaluation"})


def cheatsheet():
    return "prdtA: Reverse-Polish (postfix) evaluation"

# moirais.fn — function file (hadesllm/moirais)
"""Supervised learning paradigm: learn mapping f(x)->y from labeled examples."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_supervised_learning"]


def geron_supervised_learning(X, y):
    """
    Supervised learning paradigm: learn mapping f(x)->y from labeled examples

    Formula: minimize L(f(X), Y) over f in hypothesis class

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 1
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Supervised learning paradigm: learn mapping f(x)->y from labeled examples"})


def cheatsheet():
    return "hmsup: Supervised learning paradigm: learn mapping f(x)->y from labeled examples"

# moirais.fn — function file (hadesllm/moirais)
"""Self-supervised learning: generate labels from the data itself via pretext task."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_self_supervised"]


def geron_self_supervised(X, pretext):
    """
    Self-supervised learning: generate labels from the data itself via pretext task

    Formula: L = E[(f(x_pretext) - y_pretext)^2]

    Parameters
    ----------
    X : array-like
        Input data.
    pretext : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 1
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Self-supervised learning: generate labels from the data itself via pretext task"})


def cheatsheet():
    return "hmself: Self-supervised learning: generate labels from the data itself via pretext task"

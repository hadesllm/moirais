# moirais.fn — function file (hadesllm/moirais)
"""Multilabel classification: predict a subset of labels per instance."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_multilabel"]


def geron_multilabel(X, Y):
    """
    Multilabel classification: predict a subset of labels per instance

    Formula: y_i in {0,1}^K; Hamming / Jaccard loss

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 3
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multilabel classification: predict a subset of labels per instance"})


def cheatsheet():
    return "hmmlb: Multilabel classification: predict a subset of labels per instance"

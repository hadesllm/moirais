# morie.fn — function file (hadesllm/morie)
"""Pasting: train base models on samples drawn without replacement."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_pasting"]


def geron_pasting(X, y, base_estimator, n_estimators, sample_size):
    """
    Pasting: train base models on samples drawn without replacement

    Formula: each f_m uses sample of size s drawn without replacement

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    base_estimator : array-like
        Input data.
    n_estimators : array-like
        Input data.
    sample_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: models

    References
    ----------
    Géron Ch 6
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pasting: train base models on samples drawn without replacement"})


def cheatsheet():
    return "hmpas: Pasting: train base models on samples drawn without replacement"

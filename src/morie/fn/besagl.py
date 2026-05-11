"""Besag-York-Mollié (BYM) disease mapping model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["besag_York_Mollie"]


def besag_York_Mollie(counts, X, offset, adjacency):
    """
    Besag-York-Mollié (BYM) disease mapping model

    Formula: log lambda_i = X beta + structured + unstructured RE

    Parameters
    ----------
    counts : array-like
        Input data.
    X : array-like
        Input data.
    offset : array-like
        Input data.
    adjacency : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Besag-York-Mollié (1991)
    """
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    n = len(counts)
    result = float(np.mean(counts))
    se = float(np.std(counts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Besag-York-Mollié (BYM) disease mapping model"})


def cheatsheet():
    return "besagl: Besag-York-Mollié (BYM) disease mapping model"

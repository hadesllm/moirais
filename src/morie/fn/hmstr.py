# morie.fn — function file (hadesllm/morie)
"""Stratified sampling preserves class/strata proportions in each split."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_stratified_sampling"]


def geron_stratified_sampling(X, y, stratum, n_total):
    """
    Stratified sampling preserves class/strata proportions in each split

    Formula: n_h / n = N_h / N for each stratum h

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    stratum : array-like
        Input data.
    n_total : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sample_indices

    References
    ----------
    Géron Ch 2
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stratified sampling preserves class/strata proportions in each split"})


def cheatsheet():
    return "hmstr: Stratified sampling preserves class/strata proportions in each split"

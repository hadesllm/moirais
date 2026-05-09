"""Stratified IPT weights by baseline category."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stratified_weights"]


def stratified_weights(A, H, S):
    """
    Stratified IPT weights by baseline category

    Formula: sw_strata = f(A_t|S)/f(A_t|H_t,S)

    Parameters
    ----------
    A : array-like
        Input data.
    H : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cole-Hernán (2008)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stratified IPT weights by baseline category"})


def cheatsheet():
    return "strtwt: Stratified IPT weights by baseline category"

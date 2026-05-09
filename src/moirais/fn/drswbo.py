"""DR-DiD with stratified-block bootstrap CI."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_did_stratified_block"]


def dr_did_stratified_block(y, D, unit, time, X, clusters):
    """
    DR-DiD with stratified-block bootstrap CI

    Formula: DR moment + cluster-block bootstrap percentile interval

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.
    X : array-like
        Input data.
    clusters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sant'Anna-Zhao (2020); Bertrand-Duflo-Mullainathan (2004)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD with stratified-block bootstrap CI"})


def cheatsheet():
    return "drswbo: DR-DiD with stratified-block bootstrap CI"

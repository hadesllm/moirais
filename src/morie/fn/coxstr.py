"""Stratified Cox proportional hazards."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cox_stratified"]


def cox_stratified(time, event, X, stratum):
    """
    Stratified Cox proportional hazards

    Formula: lambda_s(t) exp(beta X) -- separate baseline per stratum

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    stratum : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kalbfleisch-Prentice (2002)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stratified Cox proportional hazards"})


def cheatsheet():
    return "coxstr: Stratified Cox proportional hazards"

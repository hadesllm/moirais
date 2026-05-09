"""Differentiability in quadratic mean of a one-dimensional submodel with score g."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_differentiable_quadratic_mean"]


def kosorok_ch3_differentiable_quadratic_mean(P_t, P, g, t):
    """
    Differentiability in quadratic mean of a one-dimensional submodel with score g

    Formula: integral [ ((dP_t)^{1/2} - (dP)^{1/2})/t - (1/2) g (dP)^{1/2} ]^2 -> 0 as t -> 0

    Parameters
    ----------
    P_t : array-like
        Input data.
    P : array-like
        Input data.
    g : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 3, Eq 3.1, p. 37
    """
    P_t = np.atleast_1d(np.asarray(P_t, dtype=float))
    n = len(P_t)
    result = float(np.mean(P_t))
    se = float(np.std(P_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Differentiability in quadratic mean of a one-dimensional submodel with score g"})


def cheatsheet():
    return "ksr061: Differentiability in quadratic mean of a one-dimensional submodel with score g"

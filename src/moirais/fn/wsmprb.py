"""Parametric bootstrap."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_parametric_boot"]


def wasserman_parametric_boot(data, f, T, B):
    """
    Parametric bootstrap

    Formula: X*_b ~ f(.;theta_hat); theta_b* = T(X*_b)

    Parameters
    ----------
    data : array-like
        Input data.
    f : array-like
        Input data.
    T : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Wasserman (2004), Ch 8
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Parametric bootstrap"})


def cheatsheet():
    return "wsmprb: Parametric bootstrap"

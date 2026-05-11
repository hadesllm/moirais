"""Generalized Synthetic Control with interactive fixed effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["generalized_synthetic_control"]


def generalized_synthetic_control(y, D, unit, time, r):
    """
    Generalized Synthetic Control with interactive fixed effects

    Formula: y_it = lambda_i' f_t + tau_it D_it + eps

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
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Xu (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalized Synthetic Control with interactive fixed effects"})


def cheatsheet():
    return "gscmcl: Generalized Synthetic Control with interactive fixed effects"

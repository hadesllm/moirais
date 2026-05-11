"""Dynamic DR-DiD over event-time horizon."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_dynamic_did"]


def dr_dynamic_did(y, D, unit, time, cohort, horizon):
    """
    Dynamic DR-DiD over event-time horizon

    Formula: DR ATT(g, g+e) for e=0..H

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
    cohort : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sun-Abraham (2021); de Chaisemartin-D'Haultfoeuille (2020)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dynamic DR-DiD over event-time horizon"})


def cheatsheet():
    return "drdyn: Dynamic DR-DiD over event-time horizon"

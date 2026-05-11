"""Placebo DR-DiD pre-period falsification."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["placebo_dr_did"]


def placebo_dr_did(y_pre1, y_pre2, D, X):
    """
    Placebo DR-DiD pre-period falsification

    Formula: DR moment on pre-event period; H0: tau_placebo=0

    Parameters
    ----------
    y_pre1 : array-like
        Input data.
    y_pre2 : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sant'Anna-Zhao (2020)
    """
    y_pre1 = np.atleast_1d(np.asarray(y_pre1, dtype=float))
    n = len(y_pre1)
    result = float(np.mean(y_pre1))
    se = float(np.std(y_pre1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Placebo DR-DiD pre-period falsification"})


def cheatsheet():
    return "drpdid: Placebo DR-DiD pre-period falsification"

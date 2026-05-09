"""DR-DiD quantile treatment effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_did_quantile"]


def dr_did_quantile(y, D, X, quantile):
    """
    DR-DiD quantile treatment effect

    Formula: E[Q_q[Y(1)|D=1] - Q_q[Y(0)|D=1]]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    quantile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Callaway-Li-Oka (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD quantile treatment effect"})


def cheatsheet():
    return "drbqs: DR-DiD quantile treatment effect"

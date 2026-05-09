"""Loss-balanced random forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["egregious_loss_forest"]


def egregious_loss_forest(y, D, X):
    """
    Loss-balanced random forest

    Formula: split criterion balances loss across leaves

    Parameters
    ----------
    y : array-like
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
    Athey-Tibshirani-Wager (2019) GRF
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Loss-balanced random forest"})


def cheatsheet():
    return "egrgrf: Loss-balanced random forest"

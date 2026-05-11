"""Minimax risk."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_minimax"]


def wasserman_minimax(loss, estimator, family):
    """
    Minimax risk

    Formula: R_minmax = inf_T sup_F R(T, F)

    Parameters
    ----------
    loss : array-like
        Input data.
    estimator : array-like
        Input data.
    family : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Wasserman (2004), Ch 12
    """
    loss = np.atleast_1d(np.asarray(loss, dtype=float))
    n = len(loss)
    result = float(np.mean(loss))
    se = float(np.std(loss, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Minimax risk"})


def cheatsheet():
    return "wsmmin: Minimax risk"

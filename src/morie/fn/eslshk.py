"""Shrinkage in boosting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_shrinkage"]


def esl_shrinkage(nu):
    """
    Shrinkage in boosting

    Formula: f_m = f_{m-1} + nu h_m, 0<nu<1

    Parameters
    ----------
    nu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: factor

    References
    ----------
    Hastie ESL Ch 10
    """
    nu = np.atleast_1d(np.asarray(nu, dtype=float))
    n = len(nu)
    result = float(np.mean(nu))
    se = float(np.std(nu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shrinkage in boosting"})


def cheatsheet():
    return "eslshk: Shrinkage in boosting"

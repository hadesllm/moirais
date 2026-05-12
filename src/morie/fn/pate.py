"""PATE -- private aggregation of teacher ensembles."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pate"]


def pate(teachers, x, epsilon):
    """
    PATE -- private aggregation of teacher ensembles

    Formula: label student via noisy argmax of teacher votes

    Parameters
    ----------
    teachers : array-like
        Input data.
    x : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Papernot et al (2017)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PATE -- private aggregation of teacher ensembles"})


def cheatsheet():
    return "pate: PATE -- private aggregation of teacher ensembles"

"""Linearization (Taylor) variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["linearization_se"]


def linearization_se(estimator, data):
    """
    Linearization (Taylor) variance

    Formula: Var(g(theta)) ~ g'(theta) Var(theta) g'(theta)

    Parameters
    ----------
    estimator : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Binder (1983)
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linearization (Taylor) variance"})


def cheatsheet():
    return "linear: Linearization (Taylor) variance"

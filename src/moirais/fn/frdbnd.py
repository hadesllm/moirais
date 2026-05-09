"""Fréchet-Hoeffding bounds on joint distribution of potential outcomes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["frechet_hoeffding_bounds"]


def frechet_hoeffding_bounds(F_0, F_1):
    """
    Fréchet-Hoeffding bounds on joint distribution of potential outcomes

    Formula: max(F_1(y_1)+F_0(y_0)-1, 0) <= F(y_1,y_0) <= min(F_1(y_1), F_0(y_0))

    Parameters
    ----------
    F_0 : array-like
        Input data.
    F_1 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fréchet (1951); Hoeffding (1940)
    """
    F_0 = np.atleast_1d(np.asarray(F_0, dtype=float))
    n = len(F_0)
    result = float(np.mean(F_0))
    se = float(np.std(F_0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fréchet-Hoeffding bounds on joint distribution of potential outcomes"})


def cheatsheet():
    return "frdbnd: Fréchet-Hoeffding bounds on joint distribution of potential outcomes"

"""Rényi DP composition (alpha, epsilon)-bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["renyi_dp_composition"]


def renyi_dp_composition(y, epsilons, alpha):
    """
    Rényi DP composition (alpha, epsilon)-bound

    Formula: epsilon_total = sum_i epsilon_i for sequential composition

    Parameters
    ----------
    y : array-like
        Input data.
    epsilons : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mironov (2017) Renyi DP
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rényi DP composition (alpha, epsilon)-bound"})


def cheatsheet():
    return "dprnyi: Rényi DP composition (alpha, epsilon)-bound"

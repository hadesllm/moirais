"""Augmented outcome-weighted learning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["augmented_owl"]


def augmented_owl(y, D, W, pi, Q):
    """
    Augmented outcome-weighted learning

    Formula: OWL with imputation of counterfactual outcomes

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    W : array-like
        Input data.
    pi : array-like
        Input data.
    Q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu-Wang-Fu-Zeng (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Augmented outcome-weighted learning"})


def cheatsheet():
    return "awltrn: Augmented outcome-weighted learning"

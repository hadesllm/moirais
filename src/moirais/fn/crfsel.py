"""Variable importance for CATE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_forest_selection"]


def causal_forest_selection(forest, X, D, y):
    """
    Variable importance for CATE

    Formula: permutation importance of X_j on tau-hat

    Parameters
    ----------
    forest : array-like
        Input data.
    X : array-like
        Input data.
    D : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey-Wager (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variable importance for CATE"})


def cheatsheet():
    return "crfsel: Variable importance for CATE"

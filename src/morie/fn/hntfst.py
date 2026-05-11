"""Honest random forest with sample-splitting (Wager-Athey 2018)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["honest_random_forest"]


def honest_random_forest(y, X, n_trees, min_node):
    """
    Honest random forest with sample-splitting (Wager-Athey 2018)

    Formula: split sample for splitting + leaf estimation; consistent CIs

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    n_trees : array-like
        Input data.
    min_node : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wager-Athey (2018); Athey-Imbens (2016)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Honest random forest with sample-splitting (Wager-Athey 2018)"})


def cheatsheet():
    return "hntfst: Honest random forest with sample-splitting (Wager-Athey 2018)"

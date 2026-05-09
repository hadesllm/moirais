"""Item-total correlation (corrected)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ctt_item_total_corr"]


def ctt_item_total_corr(X, item_index):
    """
    Item-total correlation (corrected)

    Formula: r between item and (total - item)

    Parameters
    ----------
    X : array-like
        Input data.
    item_index : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nunnally-Bernstein (1994)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(X), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Item-total correlation (corrected)"})
    result = stats.spearmanr(X[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Item-total correlation (corrected)"})


def cheatsheet():
    return "cttitc: Item-total correlation (corrected)"

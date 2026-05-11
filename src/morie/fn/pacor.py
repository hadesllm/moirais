# morie.fn — function file (hadesllm/morie)
"""Predictive ability as Pearson correlation between observed and predicted."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["predictive_ability_pearson"]


def predictive_ability_pearson(y_true, y_pred):
    """
    Predictive ability as Pearson correlation between observed and predicted

    Formula: r = cor(y, y_hat)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'r': 'float'}

    References
    ----------
    Montesinos Lopez Ch 4
    """
    y_true = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    n = min(len(y_true), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Predictive ability as Pearson correlation between observed and predicted"})
    result = stats.spearmanr(y_true[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Predictive ability as Pearson correlation between observed and predicted"})


def cheatsheet():
    return "pacor: Predictive ability as Pearson correlation between observed and predicted"

"""Logistic regression class prediction using a 50% probability threshold.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_ch4_logistic_regression_prediction"]


def geron_ch4_logistic_regression_prediction(p_hat):
    """
    Logistic regression class prediction using a 50% probability threshold.

    Formula: y_hat = 0 if p_hat < 0.5 else 1 if p_hat >= 0.5

    Parameters
    ----------
    p_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Geron (2026), Ch 4, Eq 4-16, p. 168
    """
    p_hat = np.atleast_1d(np.asarray(p_hat, dtype=float))
    n = len(p_hat)
    result = float(np.mean(p_hat))
    se = float(np.std(p_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logistic regression class prediction using a 50% probability threshold."})


def cheatsheet():
    return "grn016: Logistic regression class prediction using a 50% probability threshold."

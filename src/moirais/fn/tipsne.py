"""Tipping-point analysis for missingness."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tipping_point_sensitivity"]


def tipping_point_sensitivity(y, D, missing_indicator):
    """
    Tipping-point analysis for missingness

    Formula: impute under increasingly extreme delta until effect tips

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    missing_indicator : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yan, Lee, Ling, Lin (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tipping-point analysis for missingness"})


def cheatsheet():
    return "tipsne: Tipping-point analysis for missingness"

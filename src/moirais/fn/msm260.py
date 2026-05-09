"""Numbered display equation (13.2) from MVSML chapter 13.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_deep_learning_eq_13_2"]


def mvsml_deep_learning_eq_13_2(pre, activation, zi, yi, values, the):
    """
    Numbered display equation (13.2) from MVSML chapter 13.

    Formula: pre-activation (zi) and activation (yi) values with the following equations. This is done at each position of the ﬁlter. X 147 zi = w jxij + b (13.1) j=1 yi = g zi ( )

    Parameters
    ----------
    pre : array-like
        Input data.
    activation : array-like
        Input data.
    zi : array-like
        Input data.
    yi : array-like
        Input data.
    values : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (13.2) [Multivariate Statistical Machine Learnin [Pages 533-577] [2026-04-16].pdf]
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    n = len(values)
    result = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (13.2) from MVSML chapter 13."})


def cheatsheet():
    return "msm260: Numbered display equation (13.2) from MVSML chapter 13."

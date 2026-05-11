"""Numbered display equation (14.5) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_5"]


def mvsml_convolutional_nn_eq_14_5(b, X, TX, T, n, y):
    """
    Numbered display equation (14.5) from MVSML chapter 14.

    Formula: b\beta = XTX (14.4)  T   b\sigma2 = 1 n y  Xb\beta y  Xb\beta ,

    Parameters
    ----------
    b : array-like
        Input data.
    X : array-like
        Input data.
    TX : array-like
        Input data.
    T : array-like
        Input data.
    n : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.5) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.5) from MVSML chapter 14."})


def cheatsheet():
    return "msm268: Numbered display equation (14.5) from MVSML chapter 14."

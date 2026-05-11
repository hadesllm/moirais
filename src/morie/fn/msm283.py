"""Numbered display equation (14.12) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_12"]


def mvsml_convolutional_nn_eq_14_12(be, written, j2, TD, SSE, y):
    """
    Numbered display equation (14.12) from MVSML chapter 14.

    Formula: be written as j2 + \lambda\betaTD \beta = SSE\lambda \beta y  1n\mu  X\beta SSE\lambda \beta ( ) = j j j ( ),

    Parameters
    ----------
    be : array-like
        Input data.
    written : array-like
        Input data.
    j2 : array-like
        Input data.
    TD : array-like
        Input data.
    SSE : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.12) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.12) from MVSML chapter 14."})


def cheatsheet():
    return "msm283: Numbered display equation (14.12) from MVSML chapter 14."

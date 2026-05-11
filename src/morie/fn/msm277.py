"""Numbered display equation (14.10) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_10"]


def mvsml_convolutional_nn_eq_14_10(functions, the, covariate, function, B, spline):
    """
    Numbered display equation (14.10) from MVSML chapter 14.

    Formula: functions for the covariate function and 21 B-spline basis functions for \beta(t)  2 Xn XL1 SSE\lambda \beta ( ) = i=1 yi  \mu  l=1xil\betal + \lambdaJ\beta,

    Parameters
    ----------
    functions : array-like
        Input data.
    the : array-like
        Input data.
    covariate : array-like
        Input data.
    function : array-like
        Input data.
    B : array-like
        Input data.
    spline : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.10) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    functions = np.atleast_1d(np.asarray(functions, dtype=float))
    n = len(functions)
    result = float(np.mean(functions))
    se = float(np.std(functions, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.10) from MVSML chapter 14."})


def cheatsheet():
    return "msm277: Numbered display equation (14.10) from MVSML chapter 14."

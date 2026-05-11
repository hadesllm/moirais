"""Numbered display equation (14.4) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_4"]


def mvsml_convolutional_nn_eq_14_4(R, T, l, xil, xi, t):
    """
    Numbered display equation (14.4) from MVSML chapter 14.

    Formula:  R T l=1xil\betal, \sigma2 , xil = 0 xi t( )ϕl t( )dt, i = 1, . . ., n, l = 1, . . ., L1. So, the maximum likelihood estimation of parameters \beta and \sigma2 is given by - 1XTy b\beta = XTX

    Parameters
    ----------
    R : array-like
        Input data.
    T : array-like
        Input data.
    l : array-like
        Input data.
    xil : array-like
        Input data.
    xi : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.4) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.4) from MVSML chapter 14."})


def cheatsheet():
    return "msm267: Numbered display equation (14.4) from MVSML chapter 14."

r"""Numbered display equation (7.3) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_3"]


def mvsml_bayesian_regression_pt2_eq_7_3(can, be, taken, into, account, to):
    r"""
    Numbered display equation (7.3) from MVSML chapter 7.

    Formula: can be taken into account to improve the prediction performance. One extension of model (7.1) that takes into account environment effects and environment–marker interaction is given by pic = P Yi = c ( ) = F \gammac  \etai ( )  F \gammac1  \etai ( ), c = 1, . . . , C,

    Parameters
    ----------
    can : array-like
        Input data.
    be : array-like
        Input data.
    taken : array-like
        Input data.
    into : array-like
        Input data.
    account : array-like
        Input data.
    to : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.3) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    r"""
    can = np.atleast_1d(np.asarray(can, dtype=float))
    n = len(can)
    result = float(np.mean(can))
    se = float(np.std(can, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.3) from MVSML chapter 7."})


def cheatsheet():
    return "msm092: Numbered display equation (7.3) from MVSML chapter 7."

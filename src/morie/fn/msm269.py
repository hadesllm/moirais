r"""Numbered display equation (14.6) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_6"]


def mvsml_convolutional_nn_eq_14_6(uously, observed, Usually, it, only, measured):
    r"""
    Numbered display equation (14.6) from MVSML chapter 14.

    Formula: uously observed. Usually, it is only measured in a ﬁnite number of points t1 < t2 < . . . < tm in time or another domain. So, to complete the solution described before, the usual approach is also to assume that the covariate function can be represented as a linear combination of a set of basis functions (\psil(+), l = 1, . . ., L2) 14.1 Principles of Functional Linear Regression Analyses 581 XL2 xi t( ) = o=1cio\psio t( ),

    Parameters
    ----------
    uously : array-like
        Input data.
    observed : array-like
        Input data.
    Usually : array-like
        Input data.
    it : array-like
        Input data.
    only : array-like
        Input data.
    measured : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.6) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    r"""
    uously = np.atleast_1d(np.asarray(uously, dtype=float))
    n = len(uously)
    result = float(np.mean(uously))
    se = float(np.std(uously, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.6) from MVSML chapter 14."})


def cheatsheet():
    return "msm269: Numbered display equation (14.6) from MVSML chapter 14."

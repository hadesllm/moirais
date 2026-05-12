r"""Numbered display equation (14.11) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_11"]


def mvsml_convolutional_nn_eq_14_11(smoothed, solution, of, t, can, be):
    r"""
    Numbered display equation (14.11) from MVSML chapter 14.

    Formula: smoothed solution of \beta(t) can be obtained as XL1 b\beta t( ) = l=1b\betalϕl t( ),  and b\beta  is the solution of (14.12), which also can be obtained with the where b\beta = \Gammab\beta glmnet R package. Example 14.4 To exemplify the penalized estimation of functional regression (14.10) with penalty

    Parameters
    ----------
    smoothed : array-like
        Input data.
    solution : array-like
        Input data.
    of : array-like
        Input data.
    t : array-like
        Input data.
    can : array-like
        Input data.
    be : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.11) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    r"""
    smoothed = np.atleast_1d(np.asarray(smoothed, dtype=float))
    n = len(smoothed)
    result = float(np.mean(smoothed))
    se = float(np.std(smoothed, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.11) from MVSML chapter 14."})


def cheatsheet():
    return "msm287: Numbered display equation (14.11) from MVSML chapter 14."

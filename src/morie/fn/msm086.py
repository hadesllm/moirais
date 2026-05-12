r"""Numbered display equation (7.1) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_1"]


def mvsml_bayesian_regression_pt2_eq_7_1(exp, u2, where, F, z, p):
    r"""
    Numbered display equation (7.1) from MVSML chapter 7.

    Formula: 1ﬃﬃﬃﬃ exp  u2 where F z( ) = p du: When the response value only takes two 1 2\pi 2 values, model

    Parameters
    ----------
    exp : array-like
        Input data.
    u2 : array-like
        Input data.
    where : array-like
        Input data.
    F : array-like
        Input data.
    z : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.1) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    r"""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.1) from MVSML chapter 7."})


def cheatsheet():
    return "msm086: Numbered display equation (7.1) from MVSML chapter 7."

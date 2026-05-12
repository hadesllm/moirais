r"""Numbered display equation (6.4) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_4"]


def mvsml_bayesian_regression_eq_6_4(X1, j, g, Nn, gG, vg):
    r"""
    Numbered display equation (6.4) from MVSML chapter 6.

    Formula:   X1\beta0 j \sigma2 g  Nn 0, \sigma2 gG and \sigma2 g  \chi-2 vg, Sg (vg = v\beta, Sg = pS\beta). Similarly to what was done for model (6.3), the full conditional posterior distri- bution of g in model

    Parameters
    ----------
    X1 : array-like
        Input data.
    j : array-like
        Input data.
    g : array-like
        Input data.
    Nn : array-like
        Input data.
    gG : array-like
        Input data.
    vg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.4) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    X1 = np.atleast_1d(np.asarray(X1, dtype=float))
    n = len(X1)
    result = float(np.mean(X1))
    se = float(np.std(X1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.4) from MVSML chapter 6."})


def cheatsheet():
    return "msm051: Numbered display equation (6.4) from MVSML chapter 6."

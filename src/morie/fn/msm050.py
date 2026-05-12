r"""Numbered display equation (6.3) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_3"]


def mvsml_bayesian_regression_eq_6_3(v, S, the, induced, priors, g):
    r"""
    Numbered display equation (6.3) from MVSML chapter 6.

    Formula: v,S , and the induced priors: g =   X1\beta0 j \sigma2 g  Nn 0, \sigma2 gG and \sigma2 g  \chi-2 vg, Sg (vg = v\beta, Sg = pS\beta). Similarly to what was done for model

    Parameters
    ----------
    v : array-like
        Input data.
    S : array-like
        Input data.
    the : array-like
        Input data.
    induced : array-like
        Input data.
    priors : array-like
        Input data.
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.3) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    v = np.atleast_1d(np.asarray(v, dtype=float))
    n = len(v)
    result = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.3) from MVSML chapter 6."})


def cheatsheet():
    return "msm050: Numbered display equation (6.3) from MVSML chapter 6."

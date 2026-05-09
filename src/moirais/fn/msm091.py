"""Numbered display equation (7.1) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_1"]


def mvsml_bayesian_regression_pt2_eq_7_1(The, R, code, to, reproduce, the):
    """
    Numbered display equation (7.1) from MVSML chapter 7.

    Formula: (0.06) (0.01) (0.06) (0.01) (0.06) The R code to reproduce the results in Table 7.1 is given in Appendix 1. In some applications, additional information is available, such as the sites (loca- tions) where the experiments were conducted, environmental covariates, etc., which can be taken into account to improve the prediction performance. One extension of model

    Parameters
    ----------
    The : array-like
        Input data.
    R : array-like
        Input data.
    code : array-like
        Input data.
    to : array-like
        Input data.
    reproduce : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.1) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    The = np.atleast_1d(np.asarray(The, dtype=float))
    n = len(The)
    result = float(np.mean(The))
    se = float(np.std(The, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.1) from MVSML chapter 7."})


def cheatsheet():
    return "msm091: Numbered display equation (7.1) from MVSML chapter 7."

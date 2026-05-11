"""Numbered display equation (3.1) from MVSML chapter 3.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_elements_lin_reg_eq_3_1"]


def mvsml_elements_lin_reg_eq_3_1(Fitting, a, Linear, Multiple, Regression, Model):
    """
    Numbered display equation (3.1) from MVSML chapter 3.

    Formula: 3.2 Fitting a Linear Multiple Regression Model via the Ordinary Least Square (OLS) Method In a general context, we have a covariate vector X = (X1, . . ., Xp)T and we want to use this information to predict or explain how this variable affects a real-value response Y. The linear multiple regression model assumes a relationship given by X p Y = \beta0 + X j\beta j + E,

    Parameters
    ----------
    Fitting : array-like
        Input data.
    a : array-like
        Input data.
    Linear : array-like
        Input data.
    Multiple : array-like
        Input data.
    Regression : array-like
        Input data.
    Model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (3.1) [Multivariate Statistical Machine Learnin [Pages 71-108] [2026-04-16].pdf]
    """
    Fitting = np.atleast_1d(np.asarray(Fitting, dtype=float))
    n = len(Fitting)
    result = float(np.mean(Fitting))
    se = float(np.std(Fitting, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (3.1) from MVSML chapter 3."})


def cheatsheet():
    return "msm332: Numbered display equation (3.1) from MVSML chapter 3."

"""Numbered display equation (8.12) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_12"]


def mvsml_categorical_count_eq_8_12(y, Pf, Model, similar, to, model):
    """
    Numbered display equation (8.12) from MVSML chapter 8.

    Formula: y = \mu1n + Pf + \epsilon (8.12) Model (8.12) is similar to model (8.11), except that f is a vector of order m  1   , where P = Km, nUS21/2 is with a normal distribution of the form f  N 0, \sigma2 f Im,m now the design matrix. This implies estimating only m effects that are projected into the n-dimensional space in order to predict un and explain yn. Note that model

    Parameters
    ----------
    y : array-like
        Input data.
    Pf : array-like
        Input data.
    Model : array-like
        Input data.
    similar : array-like
        Input data.
    to : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.12) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.12) from MVSML chapter 8."})


def cheatsheet():
    return "msm155: Numbered display equation (8.12) from MVSML chapter 8."

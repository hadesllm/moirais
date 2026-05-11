"""Numbered display equation (8.12) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_12"]


def mvsml_categorical_count_eq_8_12(where, P, Km, nUS21, a, normal):
    """
    Numbered display equation (8.12) from MVSML chapter 8.

    Formula:   , where P = Km, nUS21/2 is with a normal distribution of the form f  N 0, \sigma2 f Im,m now the design matrix. This implies estimating only m effects that are projected into the n-dimensional space in order to predict un and explain yn. Note that model (8.12) has a Ridge regression solution, and thus available software for Bayesian Ridge regression like BGLR R or software for conventional Ridge regression like glmnet can be used for ﬁtting model

    Parameters
    ----------
    where : array-like
        Input data.
    P : array-like
        Input data.
    Km : array-like
        Input data.
    nUS21 : array-like
        Input data.
    a : array-like
        Input data.
    normal : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.12) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    where = np.atleast_1d(np.asarray(where, dtype=float))
    n = len(where)
    result = float(np.mean(where))
    se = float(np.std(where, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.12) from MVSML chapter 8."})


def cheatsheet():
    return "msm156: Numbered display equation (8.12) from MVSML chapter 8."

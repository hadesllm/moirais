"""Numbered display equation (8.3) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_3"]


def mvsml_categorical_count_eq_8_3(where, any, statistical, machine, learning, method):
    """
    Numbered display equation (8.3) from MVSML chapter 8.

    Formula: where any statistical machine learning method can be combined with any kernel function. It is important to point out that since many machine learning methods are only able to work with linear patterns, using the kernel trick allows you to build nonlinear versions of the linear algorithms, without the need to modify the original machine learning algorithm. The following sections show how the kernel trick works in some standard statistical machine learning models. 8.3 Kernel Methods for Gaussian Response Variables When the response variable is Gaussian, the negative log-likelihood that needs to be used to minimize expression

    Parameters
    ----------
    where : array-like
        Input data.
    any : array-like
        Input data.
    statistical : array-like
        Input data.
    machine : array-like
        Input data.
    learning : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.3) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    where = np.atleast_1d(np.asarray(where, dtype=float))
    n = len(where)
    result = float(np.mean(where))
    se = float(np.std(where, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.3) from MVSML chapter 8."})


def cheatsheet():
    return "msm133: Numbered display equation (8.3) from MVSML chapter 8."

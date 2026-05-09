"""Numbered display equation (8.3) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_3"]


def mvsml_categorical_count_eq_8_3(function, It, important, to, point, out):
    """
    Numbered display equation (8.3) from MVSML chapter 8.

    Formula: function. It is important to point out that since many machine learning methods are only able to work with linear patterns, using the kernel trick allows you to build nonlinear versions of the linear algorithms, without the need to modify the original machine learning algorithm. The following sections show how the kernel trick works in some standard statistical machine learning models. 8.3 Kernel Methods for Gaussian Response Variables When the response variable is Gaussian, the negative log-likelihood that needs to be used to minimize expression (8.3) belongs to a normal distribution and the expres- sion

    Parameters
    ----------
    function : array-like
        Input data.
    It : array-like
        Input data.
    important : array-like
        Input data.
    to : array-like
        Input data.
    point : array-like
        Input data.
    out : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.3) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    function = np.atleast_1d(np.asarray(function, dtype=float))
    n = len(function)
    result = float(np.mean(function))
    se = float(np.std(function, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.3) from MVSML chapter 8."})


def cheatsheet():
    return "msm134: Numbered display equation (8.3) from MVSML chapter 8."

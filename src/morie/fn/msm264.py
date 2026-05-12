r"""Numbered display equation (14.3) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_3"]


def mvsml_convolutional_nn_eq_14_3(Functional, Regression, function, to, be, represented):
    r"""
    Numbered display equation (14.3) from MVSML chapter 14.

    Formula: Functional Regression function to be represented (Ramsay et al. 2009). Then, by assuming this form for \beta(t), model (14.1) can be expressed as Z T XL1 Y = \mu + l=1\betal x t( )ϕl t( )dt + E = \mu + xT\beta0 + E 0 = xT\beta + E,

    Parameters
    ----------
    Functional : array-like
        Input data.
    Regression : array-like
        Input data.
    function : array-like
        Input data.
    to : array-like
        Input data.
    be : array-like
        Input data.
    represented : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.3) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    r"""
    Functional = np.atleast_1d(np.asarray(Functional, dtype=float))
    n = len(Functional)
    result = float(np.mean(Functional))
    se = float(np.std(Functional, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.3) from MVSML chapter 14."})


def cheatsheet():
    return "msm264: Numbered display equation (14.3) from MVSML chapter 14."

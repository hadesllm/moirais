"""Numbered display equation (8.13) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_13"]


def mvsml_categorical_count_eq_8_13(m, Reproducing, Kernel, Hilbert, Spaces, Regression):
    """
    Numbered display equation (8.13) from MVSML chapter 8.

    Formula: 264 599 15 32 74 m 296 8 Reproducing Kernel Hilbert Spaces Regression and Classiﬁcation Methods y = \mu1 + ZE\betaE + Pu1f + Pu2l + \epsilon,

    Parameters
    ----------
    m : array-like
        Input data.
    Reproducing : array-like
        Input data.
    Kernel : array-like
        Input data.
    Hilbert : array-like
        Input data.
    Spaces : array-like
        Input data.
    Regression : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.13) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    n = len(m)
    result = float(np.mean(m))
    se = float(np.std(m, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.13) from MVSML chapter 8."})


def cheatsheet():
    return "msm158: Numbered display equation (8.13) from MVSML chapter 8."

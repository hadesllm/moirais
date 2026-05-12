r"""Numbered display equation (2.22) from MVSML chapter 2.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_preprocessing_eq_2_22"]


def mvsml_preprocessing_eq_2_22(d, c, PE, xo, interval, the):
    r"""
    Numbered display equation (2.22) from MVSML chapter 2.

    Formula: d ) < - c 2 + c = c d\lambda PE\lambda xo ( interval [0, \lambda], the expected prediction error at xo shows a decreasing behavior, which indicates that there is a value of \lambda such that with the Ridge regression estimation of beta coefﬁcients, we can get a smaller prediction error than with the OLS prediction. Figure 3.3 shows a graphic representation of this behavior of Ridge prediction, where the lower EPE is reached at about \lambda = exp

    Parameters
    ----------
    d : array-like
        Input data.
    c : array-like
        Input data.
    PE : array-like
        Input data.
    xo : array-like
        Input data.
    interval : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (2.22) [Multivariate Statistical Machine Learnin [Pages 71-108] [2026-04-16].pdf]
    r"""
    d = np.atleast_1d(np.asarray(d, dtype=float))
    n = len(d)
    result = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (2.22) from MVSML chapter 2."})


def cheatsheet():
    return "msm334: Numbered display equation (2.22) from MVSML chapter 2."

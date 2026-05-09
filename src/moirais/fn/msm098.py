"""Numbered display equation (7.5) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_5"]


def mvsml_bayesian_regression_pt2_eq_7_5(pointed, out, Chaps, This, difference, even):
    """
    Numbered display equation (7.5) from MVSML chapter 7.

    Formula: (0.07) pointed out in Chaps. 5 and 6. This difference is even greater when the number of markers is larger than the number of observations. Example 3 Binary Traits For this example, we used the EYT Toy data set consisting of 40 lines, four environments (Bed5IR, EHT, Flat5IR, and LHT), and a response binary variable based on plant Height (0 = low, 1 = high). For this example, marker information is not available, only the genomic relationship matrix for the 40 lines. So, only the models (M3 and M4) in (7.3) and (7.4) are ﬁtted, and the performance prediction for these models was evaluated using cross-validation. Also, in this comparison model, (M5)

    Parameters
    ----------
    pointed : array-like
        Input data.
    out : array-like
        Input data.
    Chaps : array-like
        Input data.
    This : array-like
        Input data.
    difference : array-like
        Input data.
    even : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.5) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    pointed = np.atleast_1d(np.asarray(pointed, dtype=float))
    n = len(pointed)
    result = float(np.mean(pointed))
    se = float(np.std(pointed, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.5) from MVSML chapter 7."})


def cheatsheet():
    return "msm098: Numbered display equation (7.5) from MVSML chapter 7."

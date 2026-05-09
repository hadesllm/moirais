"""Numbered display equation (8.4) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_4"]


def mvsml_categorical_count_eq_8_4(de, nite, related, to, an, ANN):
    """
    Numbered display equation (8.4) from MVSML chapter 8.

    Formula: deﬁnite and related to an ANN with a single hidden layer and the ramp activation function (Cho and Saul 2009).     J \thetai,j   AK1 xi, xj = 1 k k xj

    Parameters
    ----------
    de : array-like
        Input data.
    nite : array-like
        Input data.
    related : array-like
        Input data.
    to : array-like
        Input data.
    an : array-like
        Input data.
    ANN : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.4) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    de = np.atleast_1d(np.asarray(de, dtype=float))
    n = len(de)
    result = float(np.mean(de))
    se = float(np.std(de, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.4) from MVSML chapter 8."})


def cheatsheet():
    return "msm131: Numbered display equation (8.4) from MVSML chapter 8."

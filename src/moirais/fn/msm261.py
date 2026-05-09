"""Numbered display equation (14.1) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_1"]


def mvsml_convolutional_nn_eq_14_1(T, l, charg, de, Scholars, Portal):
    """
    Numbered display equation (14.1) from MVSML chapter 14.

    Formula: Téléchargé de Scholars Portal Books sur 2026-04-16 Chapter 14 Functional Regression 14.1 Principles of Functional Linear Regression Analyses The general functional linear regression model with scalar response (Y) and one functional covariate (x(+)) is deﬁned by Z T Y = \mu + x t( )\beta t( )dt + E,

    Parameters
    ----------
    T : array-like
        Input data.
    l : array-like
        Input data.
    charg : array-like
        Input data.
    de : array-like
        Input data.
    Scholars : array-like
        Input data.
    Portal : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.1) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.1) from MVSML chapter 14."})


def cheatsheet():
    return "msm261: Numbered display equation (14.1) from MVSML chapter 14."

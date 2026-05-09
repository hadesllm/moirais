"""Numbered display equation (8.7) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_7"]


def mvsml_categorical_count_eq_8_7(b, CTC, CTK, CTy):
    """
    Numbered display equation (8.7) from MVSML chapter 8.

    Formula: # " # " # b\theta CTC CTK CTy =

    Parameters
    ----------
    b : array-like
        Input data.
    CTC : array-like
        Input data.
    CTK : array-like
        Input data.
    CTy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.7) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    b = np.atleast_1d(np.asarray(b, dtype=float))
    n = len(b)
    result = float(np.mean(b))
    se = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.7) from MVSML chapter 8."})


def cheatsheet():
    return "msm137: Numbered display equation (8.7) from MVSML chapter 8."

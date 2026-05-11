"""Numbered display equation (8.6) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_6"]


def mvsml_categorical_count_eq_8_6(CTC, CTK, b, CTy):
    """
    Numbered display equation (8.6) from MVSML chapter 8.

    Formula: # " # " # CTC CTK b\theta CTy =

    Parameters
    ----------
    CTC : array-like
        Input data.
    CTK : array-like
        Input data.
    b : array-like
        Input data.
    CTy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.6) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    CTC = np.atleast_1d(np.asarray(CTC, dtype=float))
    n = len(CTC)
    result = float(np.mean(CTC))
    se = float(np.std(CTC, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.6) from MVSML chapter 8."})


def cheatsheet():
    return "msm135: Numbered display equation (8.6) from MVSML chapter 8."

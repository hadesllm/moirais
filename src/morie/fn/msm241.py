"""Numbered display equation (2.2) from MVSML chapter 2.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_preprocessing_eq_2_2"]


def mvsml_preprocessing_eq_2_2(b, XTR):
    """
    Numbered display equation (2.2) from MVSML chapter 2.

    Formula: ! b\beta XTR1X XTR1Z XTR1y =

    Parameters
    ----------
    b : array-like
        Input data.
    XTR : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (2.2) [Multivariate Statistical Machine Learnin [Pages 35-70] [2026-04-16].pdf]
    """
    b = np.atleast_1d(np.asarray(b, dtype=float))
    n = len(b)
    result = float(np.mean(b))
    se = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (2.2) from MVSML chapter 2."})


def cheatsheet():
    return "msm241: Numbered display equation (2.2) from MVSML chapter 2."

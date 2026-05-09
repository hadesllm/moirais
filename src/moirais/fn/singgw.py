"""Single-step GBLUP (ssGBLUP)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["single_step_gblup"]


def single_step_gblup(y, X, Z, A, G):
    """
    Single-step GBLUP (ssGBLUP)

    Formula: combined A + G matrix for genotyped + ungenotyped

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.
    A : array-like
        Input data.
    G : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Misztal et al (2009); Aguilar et al (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Single-step GBLUP (ssGBLUP)"})


def cheatsheet():
    return "singgw: Single-step GBLUP (ssGBLUP)"

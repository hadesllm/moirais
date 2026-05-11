"""Numbered display equation (15.1) from MVSML chapter 15.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_functional_regression_eq_15_1"]


def mvsml_functional_regression_eq_15_1(Mathlouthi, et, al, through, are, given):
    """
    Numbered display equation (15.1) from MVSML chapter 15.

    Formula: (Mathlouthi et al. 2019) through \mu and \theta are given by nonparametric link functions like   \theta log \mu ( ) = f \mu x ( ) and log = f \theta x ( ),

    Parameters
    ----------
    Mathlouthi : array-like
        Input data.
    et : array-like
        Input data.
    al : array-like
        Input data.
    through : array-like
        Input data.
    are : array-like
        Input data.
    given : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (15.1) [Multivariate Statistical Machine Learnin [Pages 633-681] [2026-04-16].pdf]
    """
    Mathlouthi = np.atleast_1d(np.asarray(Mathlouthi, dtype=float))
    n = len(Mathlouthi)
    result = float(np.mean(Mathlouthi))
    se = float(np.std(Mathlouthi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (15.1) from MVSML chapter 15."})


def cheatsheet():
    return "msm323: Numbered display equation (15.1) from MVSML chapter 15."

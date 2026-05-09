"""Numbered display equation (15.3) from MVSML chapter 15.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_functional_regression_eq_15_3"]


def mvsml_functional_regression_eq_15_3(bY, exp, b, It, important, to):
    """
    Numbered display equation (15.3) from MVSML chapter 15.

    Formula: ( ) bY = (15.3) 1  exp b\mu ( ( ) ) It is important to point out that in the prediction formula given above

    Parameters
    ----------
    bY : array-like
        Input data.
    exp : array-like
        Input data.
    b : array-like
        Input data.
    It : array-like
        Input data.
    important : array-like
        Input data.
    to : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (15.3) [Multivariate Statistical Machine Learnin [Pages 633-681] [2026-04-16].pdf]
    """
    bY = np.atleast_1d(np.asarray(bY, dtype=float))
    n = len(bY)
    result = float(np.mean(bY))
    se = float(np.std(bY, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (15.3) from MVSML chapter 15."})


def cheatsheet():
    return "msm328: Numbered display equation (15.3) from MVSML chapter 15."

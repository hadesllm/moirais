"""Numbered display equation (15.4) from MVSML chapter 15.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_functional_regression_eq_15_4"]


def mvsml_functional_regression_eq_15_4(It, important, to, point, out, that):
    """
    Numbered display equation (15.4) from MVSML chapter 15.

    Formula: ) ) It is important to point out that in the prediction formula given above (15.3), (bY) is equal to the mean of the ZAP model, while under the ZAPC_RF, the predictions are obtained as 8 < b\theta > 0:5 0, bY =

    Parameters
    ----------
    It : array-like
        Input data.
    important : array-like
        Input data.
    to : array-like
        Input data.
    point : array-like
        Input data.
    out : array-like
        Input data.
    that : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (15.4) [Multivariate Statistical Machine Learnin [Pages 633-681] [2026-04-16].pdf]
    """
    It = np.atleast_1d(np.asarray(It, dtype=float))
    n = len(It)
    result = float(np.mean(It))
    se = float(np.std(It, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (15.4) from MVSML chapter 15."})


def cheatsheet():
    return "msm329: Numbered display equation (15.4) from MVSML chapter 15."

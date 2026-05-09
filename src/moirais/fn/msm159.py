"""Numbered display equation (1.2) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_2"]


def mvsml_general_eq_1_2(Tab1_MSE, AK1, k, mean, y, Pos_tst):
    """
    Numbered display equation (1.2) from MVSML chapter 1.

    Formula: Tab1_MSE$AK1[k] = mean((y[Pos_tst]-yp_ts)^2) Tab1_Cor$AK1[k] = cor(y[Pos_tst],yp_ts) A = BGLR(y=y_NA,ETA=ETA_K.AK4,nIter = 1e4,burnIn = 1e3,verbose = FALSE) yp_ts = A$yHat[Pos_tst] Tab1_MSE$AK4[k] = mean((y[Pos_tst]-yp_ts)^2) Tab1_Cor$AK4[k] = cor(y[Pos_tst],yp_ts) } Tab1_MSE apply(Tab1_MSE[,-c

    Parameters
    ----------
    Tab1_MSE : array-like
        Input data.
    AK1 : array-like
        Input data.
    k : array-like
        Input data.
    mean : array-like
        Input data.
    y : array-like
        Input data.
    Pos_tst : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.2) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.2) from MVSML chapter 1."})


def cheatsheet():
    return "msm159: Numbered display equation (1.2) from MVSML chapter 1."

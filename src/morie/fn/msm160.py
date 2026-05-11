"""Numbered display equation (1.2) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_2"]


def mvsml_general_eq_1_2(yp_ts, apply, Probs, which, max, Tab1_PCCC):
    """
    Numbered display equation (1.2) from MVSML chapter 1.

    Formula: yp_ts = apply(Probs,1,which.max)-1 Tab1_PCCC$AK1[k] = 1-mean(y[Pos_tst]!=yp_ts) A = BGLR(y=y_NA,ETA=ETA_K.AK4,response_type="ordinal",nIter = 1e4, burnIn = 1e3,verbose = FALSE) Probs = A$probs[Pos_tst,] yp_ts = apply(Probs,1,which.max)-1 Tab1_PCCC$AK4[k] = 1-mean(y[Pos_tst]!=yp_ts) } Tab1_PCCC apply(Tab1_PCCC[,-c

    Parameters
    ----------
    yp_ts : array-like
        Input data.
    apply : array-like
        Input data.
    Probs : array-like
        Input data.
    which : array-like
        Input data.
    max : array-like
        Input data.
    Tab1_PCCC : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.2) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    yp_ts = np.atleast_1d(np.asarray(yp_ts, dtype=float))
    n = len(yp_ts)
    result = float(np.mean(yp_ts))
    se = float(np.std(yp_ts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.2) from MVSML chapter 1."})


def cheatsheet():
    return "msm160: Numbered display equation (1.2) from MVSML chapter 1."

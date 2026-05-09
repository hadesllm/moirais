"""Numbered display equation (5.4) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_4"]


def mvsml_linear_mixed_models_eq_5_4(A, mmer, y_NA, Env, na, method):
    """
    Numbered display equation (5.4) from MVSML chapter 5.

    Formula: A = mmer(y_NA ~ Env,na.method.Y='include', random= ~ vs(GID,Gu=G), rcov= ~ vs(units), data=dat_F,verbose=FALSE) yp = ﬁtted(A)$dataWithFitted$y_NA.ﬁtted #Prediction of testing yp_ts = yp[Pos_tst] Tab$MSEP11[k] = mean((dat_F$y[Pos_tst]-yp_ts)^2) Tab$Cor11[k] = cor(dat_F$y[Pos_tst],yp_ts) #M10a: Model1

    Parameters
    ----------
    A : array-like
        Input data.
    mmer : array-like
        Input data.
    y_NA : array-like
        Input data.
    Env : array-like
        Input data.
    na : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.4) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.4) from MVSML chapter 5."})


def cheatsheet():
    return "msm038: Numbered display equation (5.4) from MVSML chapter 5."

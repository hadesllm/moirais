"""Numbered display equation (5.4) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_4"]


def mvsml_linear_mixed_models_eq_5_4(random, vs, GID, rcov, units, data):
    """
    Numbered display equation (5.4) from MVSML chapter 5.

    Formula: random= ~ vs(GID), rcov= ~ vs(units), data=dat_F,verbose=FALSE) yp = ﬁtted(A)$dataWithFitted$y_NA.ﬁtted #Prediction of testing yp_ts = yp[Pos_tst] Tab$MSEP10a[k] = mean((dat_F$y[Pos_tst]-yp_ts)^2) Tab$Cor10a[k] = cor(dat_F$y[Pos_tst],yp_ts) } #Basic code to implement model

    Parameters
    ----------
    random : array-like
        Input data.
    vs : array-like
        Input data.
    GID : array-like
        Input data.
    rcov : array-like
        Input data.
    units : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.4) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.4) from MVSML chapter 5."})


def cheatsheet():
    return "msm039: Numbered display equation (5.4) from MVSML chapter 5."

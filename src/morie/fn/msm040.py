"""Numbered display equation (5.4) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_4"]


def mvsml_linear_mixed_models_eq_5_4(yp_ts, yp, Pos_tst, Tab, MSEP10a, k):
    """
    Numbered display equation (5.4) from MVSML chapter 5.

    Formula: yp_ts = yp[Pos_tst] Tab$MSEP10a[k] = mean((dat_F$y[Pos_tst]-yp_ts)^2) Tab$Cor10a[k] = cor(dat_F$y[Pos_tst],yp_ts) } #Basic code to implement model (5.4) with an unstructured form for Sigma_E A = mmer(y~ Env, na.method.Y='include', random= ~ vs(GID,Gu=G)+vs(us(Env),GID,Gu=G), rcov= ~ vs(units), data=dat_F) #Basic code to implement model

    Parameters
    ----------
    yp_ts : array-like
        Input data.
    yp : array-like
        Input data.
    Pos_tst : array-like
        Input data.
    Tab : array-like
        Input data.
    MSEP10a : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.4) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    yp_ts = np.atleast_1d(np.asarray(yp_ts, dtype=float))
    n = len(yp_ts)
    result = float(np.mean(yp_ts))
    se = float(np.std(yp_ts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.4) from MVSML chapter 5."})


def cheatsheet():
    return "msm040: Numbered display equation (5.4) from MVSML chapter 5."

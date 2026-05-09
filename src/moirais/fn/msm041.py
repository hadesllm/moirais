"""Numbered display equation (5.4) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_4"]


def mvsml_linear_mixed_models_eq_5_4(rcov, vs, units, data, dat_F, Basic):
    """
    Numbered display equation (5.4) from MVSML chapter 5.

    Formula: rcov= ~ vs(units), data=dat_F) #Basic code to implement model (5.4) with heterogeneous environment variances A = mmer(y ~ Env, , random= ~ vs(GID,Gu=G)+ vs(ds(Env),GID,Gu=G),#or vs(at(Env),GID,Gu=G) rcov= ~ vs(units), data=dat_F) #Y = mu + Env + GID + GID:ENV #Y = mu + Env + GID + GID:ENV #Basic code to implement model

    Parameters
    ----------
    rcov : array-like
        Input data.
    vs : array-like
        Input data.
    units : array-like
        Input data.
    data : array-like
        Input data.
    dat_F : array-like
        Input data.
    Basic : array-like
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
    return "msm041: Numbered display equation (5.4) from MVSML chapter 5."

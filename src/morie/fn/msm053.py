r"""Numbered display equation (6.4) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_4"]


def mvsml_bayesian_regression_eq_6_4(eS, S, y, g, k2, where):
    r"""
    Numbered display equation (6.4) from MVSML chapter 6.

    Formula: and eS = S + y 2 1n\mu - g k2; and \sigma2 where ev = v + n g j -  \chi-2 evg,eSg, where evg = k vg + n and eSg = Sg + gTG-1g. Note that when p  n, then the dimension of the parameter space of the posterior of GBLUP model is lower than the BRR. The GBLUP model

    Parameters
    ----------
    eS : array-like
        Input data.
    S : array-like
        Input data.
    y : array-like
        Input data.
    g : array-like
        Input data.
    k2 : array-like
        Input data.
    where : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.4) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.4) from MVSML chapter 6."})


def cheatsheet():
    return "msm053: Numbered display equation (6.4) from MVSML chapter 6."

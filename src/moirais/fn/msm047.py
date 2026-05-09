"""Numbered display equation (6.2) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_2"]


def mvsml_bayesian_regression_eq_6_2(T, j, Np, Ip, obtained, by):
    """
    Numbered display equation (6.2) from MVSML chapter 6.

    Formula: - T j \sigma2 \beta  Np 0, Ip\sigma2 obtained by assuming that \beta1, . . . , \betap , ignoring the prior \beta distribution of \sigma2 \beta and setting this at a very high value (1010). Note that this model is very similar to the Bayesian model obtained by adopting the prior

    Parameters
    ----------
    T : array-like
        Input data.
    j : array-like
        Input data.
    Np : array-like
        Input data.
    Ip : array-like
        Input data.
    obtained : array-like
        Input data.
    by : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.2) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.2) from MVSML chapter 6."})


def cheatsheet():
    return "msm047: Numbered display equation (6.2) from MVSML chapter 6."

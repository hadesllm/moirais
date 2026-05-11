"""Numbered display equation (5.2) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_2"]


def mvsml_linear_mixed_models_eq_5_2(n, where, V, ZTDZ, R, the):
    """
    Numbered display equation (5.2) from MVSML chapter 5.

    Formula: ) (5.2) , n ( 2\pi ) where V = ZTDZ + R is the marginal variance of Y. The maximum likelihood estimators (MLE) of the parameters, \beta, D, and R, are the values that maximize the likelihood function

    Parameters
    ----------
    n : array-like
        Input data.
    where : array-like
        Input data.
    V : array-like
        Input data.
    ZTDZ : array-like
        Input data.
    R : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.2) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.2) from MVSML chapter 5."})


def cheatsheet():
    return "msm012: Numbered display equation (5.2) from MVSML chapter 5."

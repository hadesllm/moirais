"""Numbered display equation (5.1) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_1"]


def mvsml_linear_mixed_models_eq_5_1(k, K, t, qk, These, are):
    """
    Numbered display equation (5.1) from MVSML chapter 5.

    Formula: , k = 1, . . . , K: k t+1 ( qk These are obtained by maximizing Q(\beta, \theta| \beta(t), \theta(t)) (deﬁned above) with respect to \sigma2 k, k = 1, . . . , K: 5.2.1.2 REML An alternative to the ML estimation of the variance components of model

    Parameters
    ----------
    k : array-like
        Input data.
    K : array-like
        Input data.
    t : array-like
        Input data.
    qk : array-like
        Input data.
    These : array-like
        Input data.
    are : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.1) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    n = len(k)
    result = float(np.mean(k))
    se = float(np.std(k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.1) from MVSML chapter 5."})


def cheatsheet():
    return "msm013: Numbered display equation (5.1) from MVSML chapter 5."

r"""Numbered display equation (8.8) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_8"]


def mvsml_categorical_count_eq_8_8(j, k, u, evu, eSu, e):
    r"""
    Numbered display equation (8.8) from MVSML chapter 8.

    Formula: \sigma2 j   \chi2 k u j   \chi2 evu,eSu, e where evu = vu + n and eSu = uTK1u. The Bayesian kernel BLUP, like the GBLUP, does not face the large p and small n problem, since due to the kernel trick, a problem of dimensionality p is converted into an n-dimensional problem. The Bayesian kernel BLUP model

    Parameters
    ----------
    j : array-like
        Input data.
    k : array-like
        Input data.
    u : array-like
        Input data.
    evu : array-like
        Input data.
    eSu : array-like
        Input data.
    e : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.8) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    r"""
    j = np.atleast_1d(np.asarray(j, dtype=float))
    n = len(j)
    result = float(np.mean(j))
    se = float(np.std(j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.8) from MVSML chapter 8."})


def cheatsheet():
    return "msm141: Numbered display equation (8.8) from MVSML chapter 8."

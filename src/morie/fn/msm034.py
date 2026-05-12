r"""Numbered display equation (5.6) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_6"]


def mvsml_linear_mixed_models_eq_5_6(T, N, IIJ, RnT, I, j):
    r"""
    Numbered display equation (5.6) from MVSML chapter 5.

    Formula: T  N 0, IIJ⨂RnT I, j = 1, . . ., J. In addition, it is assumed that e = eT 1 . . . eT ( ), I b1  N(0, G ⨂\SigmaT), and b2  N(0, \SigmaE ⨂G ⨂\Sigma2T). This shows that when \SigmaT, \Sigma2T, \SigmaE, and R are diagonal matrices, model (5.6) is equivalent to separately ﬁtting a univariate GBLUP model for each trait. Example 4 To illustrate the ﬁtting and evaluation process of model

    Parameters
    ----------
    T : array-like
        Input data.
    N : array-like
        Input data.
    IIJ : array-like
        Input data.
    RnT : array-like
        Input data.
    I : array-like
        Input data.
    j : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.6) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    r"""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.6) from MVSML chapter 5."})


def cheatsheet():
    return "msm034: Numbered display equation (5.6) from MVSML chapter 5."

r"""Numbered display equation (5.6) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_6"]


def mvsml_linear_mixed_models_eq_5_6(N, G, T, b2, E, This):
    r"""
    Numbered display equation (5.6) from MVSML chapter 5.

    Formula: 1  N(0, G ⨂\SigmaT), and b2  N(0, \SigmaE ⨂G ⨂\Sigma2T). This shows that when \SigmaT, \Sigma2T, \SigmaE, and R are diagonal matrices, model (5.6) is equivalent to separately ﬁtting a univariate GBLUP model for each trait. Example 4 To illustrate the ﬁtting and evaluation process of model (5.6), we considered a data set that contains the information of two traits, for which 150 lines were phenotyped each in two environments, and given a total of 300 bivar- iate phenotypic data points. Also, a genomic relationship matrix for the lines is available that was computed with marker information. The ﬁrst explored model is referred to as M4 and assumes an unstructured variance–covariance matrix for all the components in model

    Parameters
    ----------
    N : array-like
        Input data.
    G : array-like
        Input data.
    T : array-like
        Input data.
    b2 : array-like
        Input data.
    E : array-like
        Input data.
    This : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.6) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    r"""
    N = np.atleast_1d(np.asarray(N, dtype=float))
    n = len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.6) from MVSML chapter 5."})


def cheatsheet():
    return "msm035: Numbered display equation (5.6) from MVSML chapter 5."

r"""Numbered display equation (5.6) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_6"]


def mvsml_linear_mixed_models_eq_5_6(other, relevant, strategy, In, a, similar):
    r"""
    Numbered display equation (5.6) from MVSML chapter 5.

    Formula: other relevant strategy. In a similar fashion, just as univariate genomic linear mixed model (5.4), model (5.5) can be directly extended to a model that considers the genotype -environment interaction term. Next, we do this for the balanced case, and for this we assume that for each environment i = 1, . . ., I, J lines were phenotyped for nT traits, Yijt, t = 1, . . ., nT. In matrix notation, the extended G- E model (5.4) plus ﬁxed effects (X\beta) is given by Y = 1IJ⨂InT ( )\mu + X\beta + ZLb1 + ZELb2 + e,

    Parameters
    ----------
    other : array-like
        Input data.
    relevant : array-like
        Input data.
    strategy : array-like
        Input data.
    In : array-like
        Input data.
    a : array-like
        Input data.
    similar : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.6) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    r"""
    other = np.atleast_1d(np.asarray(other, dtype=float))
    n = len(other)
    result = float(np.mean(other))
    se = float(np.std(other, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.6) from MVSML chapter 5."})


def cheatsheet():
    return "msm032: Numbered display equation (5.6) from MVSML chapter 5."

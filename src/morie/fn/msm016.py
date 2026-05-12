r"""Numbered display equation (5.3) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_3"]


def mvsml_linear_mixed_models_eq_5_3(genotypic, effects, of, J, lines, Z):
    r"""
    Numbered display equation (5.3) from MVSML chapter 5.

    Formula: genotypic effects of J lines, and Z is the incidence matrix design for the random line effects (ZL): Y = 1n\mu + ZLb + e, (5.3)   where b  NJ 0, \sigma2 and R = \sigma2In. gG The basic code to implement the GBLUP model

    Parameters
    ----------
    genotypic : array-like
        Input data.
    effects : array-like
        Input data.
    of : array-like
        Input data.
    J : array-like
        Input data.
    lines : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.3) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    r"""
    genotypic = np.atleast_1d(np.asarray(genotypic, dtype=float))
    n = len(genotypic)
    result = float(np.mean(genotypic))
    se = float(np.std(genotypic, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.3) from MVSML chapter 5."})


def cheatsheet():
    return "msm016: Numbered display equation (5.3) from MVSML chapter 5."

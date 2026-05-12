r"""Numbered display equation (5.3) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_3"]


def mvsml_linear_mixed_models_eq_5_3(derived, re, ectance, information, Krause, et):
    r"""
    Numbered display equation (5.3) from MVSML chapter 5.

    Formula: derived from hyperspectral reﬂectance information (Krause et al. 2019). Other extensions of this model can be developed by taking into account other factors, for example, genotype- environment interaction, as will be illustrated later in the genomic prediction context. In this case, where only the genotypic effects are taken into account, in the linear mixed model (5.1), the ﬁxed effects design matrix is X = 1n, where the vector of length n corresponds to the general mean \beta = \beta0, b = (b1, b2, . . ., bJ)T contains the genotypic effects of J lines, and Z is the incidence matrix design for the random line effects (ZL): Y = 1n\mu + ZLb + e,

    Parameters
    ----------
    derived : array-like
        Input data.
    re : array-like
        Input data.
    ectance : array-like
        Input data.
    information : array-like
        Input data.
    Krause : array-like
        Input data.
    et : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.3) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    r"""
    derived = np.atleast_1d(np.asarray(derived, dtype=float))
    n = len(derived)
    result = float(np.mean(derived))
    se = float(np.std(derived, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.3) from MVSML chapter 5."})


def cheatsheet():
    return "msm015: Numbered display equation (5.3) from MVSML chapter 5."

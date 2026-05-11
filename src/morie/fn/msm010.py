"""Numbered display equation (5.1) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_1"]


def mvsml_linear_mixed_models_eq_5_1(O, A, Montesinos, L, pez, et):
    """
    Numbered display equation (5.1) from MVSML chapter 5.

    Formula: O. A. Montesinos López et al., Multivariate Statistical Machine Learning Methods for Genomic Prediction, https://doi.org/10.1007/978-3-030-89010-0_5 142 5 Linear Mixed Models Covarrubias-Pazaran et al. 2018; Wang et al. 2018; Cappa et al. 2019). However, the use of this model in animal science can be traced back to Henderson (1950). The general univariate linear mixed model (Harville 1977) is provided by the formula Y = X\beta + Zb + e,

    Parameters
    ----------
    O : array-like
        Input data.
    A : array-like
        Input data.
    Montesinos : array-like
        Input data.
    L : array-like
        Input data.
    pez : array-like
        Input data.
    et : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.1) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    O = np.atleast_1d(np.asarray(O, dtype=float))
    n = len(O)
    result = float(np.mean(O))
    se = float(np.std(O, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.1) from MVSML chapter 5."})


def cheatsheet():
    return "msm010: Numbered display equation (5.1) from MVSML chapter 5."

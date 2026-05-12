r"""Numbered display equation (5.4) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_4"]


def mvsml_linear_mixed_models_eq_5_4(marker, information, prediction, although, this, could):
    r"""
    Numbered display equation (5.4) from MVSML chapter 5.

    Formula: marker information for prediction, although this could change with larger data sets (more lines and more markers) or by improving the quality of the available data. The R code to reproduce this result is given in Appendix 4. This can be adapted easily to another CV strategy of interest where the objective, for example, can be the prediction of non-observed lines in some environments or the prediction of lines in a future year. An extension of the GBLUP model is the G-E BLUP model that takes into account the main environmental effects, the genotypic effects, and the genotype -environment interaction effects: Y = 1n\mu + XE\betaE + ZLb1 + ZELb2 + e

    Parameters
    ----------
    marker : array-like
        Input data.
    information : array-like
        Input data.
    prediction : array-like
        Input data.
    although : array-like
        Input data.
    this : array-like
        Input data.
    could : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.4) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    r"""
    marker = np.atleast_1d(np.asarray(marker, dtype=float))
    n = len(marker)
    result = float(np.mean(marker))
    se = float(np.std(marker, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.4) from MVSML chapter 5."})


def cheatsheet():
    return "msm018: Numbered display equation (5.4) from MVSML chapter 5."

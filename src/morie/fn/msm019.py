"""Numbered display equation (5.1) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_1"]


def mvsml_linear_mixed_models_eq_5_1(The, R, code, to, reproduce, this):
    """
    Numbered display equation (5.1) from MVSML chapter 5.

    Formula: The R code to reproduce this result is given in Appendix 4. This can be adapted easily to another CV strategy of interest where the objective, for example, can be the prediction of non-observed lines in some environments or the prediction of lines in a future year. An extension of the GBLUP model is the G-E BLUP model that takes into account the main environmental effects, the genotypic effects, and the genotype -environment interaction effects: Y = 1n\mu + XE\betaE + ZLb1 + ZELb2 + e (5.4) where now the ﬁxed effects are part of the linear mixed model

    Parameters
    ----------
    The : array-like
        Input data.
    R : array-like
        Input data.
    code : array-like
        Input data.
    to : array-like
        Input data.
    reproduce : array-like
        Input data.
    this : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.1) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    The = np.atleast_1d(np.asarray(The, dtype=float))
    n = len(The)
    result = float(np.mean(The))
    se = float(np.std(The, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.1) from MVSML chapter 5."})


def cheatsheet():
    return "msm019: Numbered display equation (5.1) from MVSML chapter 5."

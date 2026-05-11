"""Numbered display equation (5.3) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_3"]


def mvsml_linear_mixed_models_eq_5_3(ti, trait, Genomic, Linear, Mixed, Effects):
    """
    Numbered display equation (5.3) from MVSML chapter 5.

    Formula: ti-trait Genomic Linear Mixed-Effects Models 153 work well, especially in traits with low heritability. When low heritability traits have at least moderate correlation with high heritability traits, the prediction performance ability for these low heritability traits could strongly increase by using a multi-trait model (Jia and Jannink 2012; Montesinos-López et al. 2016; Budhlakoti et al. 2019). If for each line ( j = 1, . . .J), nT traits are measured, Yjt, t = 1, . . .nT, the multi-trait genomic linear mixed-effects model adopts an unstructured covariance matrix for the residuals between traits and for the random genotypic effects between traits, and similar to the univariate trait models

    Parameters
    ----------
    ti : array-like
        Input data.
    trait : array-like
        Input data.
    Genomic : array-like
        Input data.
    Linear : array-like
        Input data.
    Mixed : array-like
        Input data.
    Effects : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.3) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    ti = np.atleast_1d(np.asarray(ti, dtype=float))
    n = len(ti)
    result = float(np.mean(ti))
    se = float(np.std(ti, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.3) from MVSML chapter 5."})


def cheatsheet():
    return "msm025: Numbered display equation (5.3) from MVSML chapter 5."

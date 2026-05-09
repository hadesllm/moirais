"""Numbered display equation (5.1) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_1"]


def mvsml_linear_mixed_models_eq_5_1(are, assumed, identically, distributed, R, where):
    """
    Numbered display equation (5.1) from MVSML chapter 5.

    Formula: are assumed as independently and identically distributed, R = \sigma2In, where n is the total number of observations. In this case, the resultant model is known as the GBLUP model and when the pedigree is used, it is referred to as PBLUP. Another kind of information between lines can also be used, such as the relationship matrices derived from hyperspectral reﬂectance information (Krause et al. 2019). Other extensions of this model can be developed by taking into account other factors, for example, genotype- environment interaction, as will be illustrated later in the genomic prediction context. In this case, where only the genotypic effects are taken into account, in the linear mixed model

    Parameters
    ----------
    are : array-like
        Input data.
    assumed : array-like
        Input data.
    identically : array-like
        Input data.
    distributed : array-like
        Input data.
    R : array-like
        Input data.
    where : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.1) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    are = np.atleast_1d(np.asarray(are, dtype=float))
    n = len(are)
    result = float(np.mean(are))
    se = float(np.std(are, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.1) from MVSML chapter 5."})


def cheatsheet():
    return "msm014: Numbered display equation (5.1) from MVSML chapter 5."

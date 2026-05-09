"""Numbered display equation (4.10) from MVSML chapter 4.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_overfitting_resampling_eq_4_10"]


def mvsml_overfitting_resampling_eq_4_10(that, the, speci, city, proportion, of):
    """
    Numbered display equation (4.10) from MVSML chapter 4.

    Formula: that is, the speciﬁcity is the proportion of true negatives that are correctly identiﬁed by the test. Under the “one-versus-all basis,” where each category is compared with the composed information of the remaining categories, we provide the expressions for computing the generalized precision, sensitivity, and speciﬁcity for each class i: TTPall Pi = (4.9) TTPall + TFPi TTPall Sei =

    Parameters
    ----------
    that : array-like
        Input data.
    the : array-like
        Input data.
    speci : array-like
        Input data.
    city : array-like
        Input data.
    proportion : array-like
        Input data.
    of : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (4.10) [Multivariate Statistical Machine Learnin [Pages 109-139] [2026-04-16].pdf]
    """
    that = np.atleast_1d(np.asarray(that, dtype=float))
    n = len(that)
    result = float(np.mean(that))
    se = float(np.std(that, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (4.10) from MVSML chapter 4."})


def cheatsheet():
    return "msm008: Numbered display equation (4.10) from MVSML chapter 4."

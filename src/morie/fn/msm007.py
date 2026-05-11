"""Numbered display equation (4.9) from MVSML chapter 4.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_overfitting_resampling_eq_4_9"]


def mvsml_overfitting_resampling_eq_4_9(proportion, of, true, positives, that, are):
    """
    Numbered display equation (4.9) from MVSML chapter 4.

    Formula: proportion of true positives that are correctly identiﬁed by the test. The precision is the proportion of correct classiﬁcation of our statistical machine learning model and represents the proportion of cases correctly classiﬁed, while the speciﬁcity is the ability of our statistical machine learning model to classify the true negative cases, that is, the speciﬁcity is the proportion of true negatives that are correctly identiﬁed by the test. Under the “one-versus-all basis,” where each category is compared with the composed information of the remaining categories, we provide the expressions for computing the generalized precision, sensitivity, and speciﬁcity for each class i: TTPall Pi =

    Parameters
    ----------
    proportion : array-like
        Input data.
    of : array-like
        Input data.
    true : array-like
        Input data.
    positives : array-like
        Input data.
    that : array-like
        Input data.
    are : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (4.9) [Multivariate Statistical Machine Learnin [Pages 109-139] [2026-04-16].pdf]
    """
    proportion = np.atleast_1d(np.asarray(proportion, dtype=float))
    n = len(proportion)
    result = float(np.mean(proportion))
    se = float(np.std(proportion, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (4.9) from MVSML chapter 4."})


def cheatsheet():
    return "msm007: Numbered display equation (4.9) from MVSML chapter 4."

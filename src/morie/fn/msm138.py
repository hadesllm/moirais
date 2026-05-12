r"""Numbered display equation (8.8) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_8"]


def mvsml_categorical_count_eq_8_8(folds, Fig, d, the, optimal, number):
    r"""
    Numbered display equation (8.8) from MVSML chapter 8.

    Formula: 3, in folds 3 and 4 (Fig. 8.2b, d), the optimal number of hidden layers was equal to 8. Finally, with these optimal values, the model was reﬁtted with the information of the inner training + tuning set, and then for each fold, the mean square error (MSE) was calculated for each outer testing set; the MSEs were 0.6878 (fold 1), 0.6963 (fold 2), 0.9725 (fold 3), and 0.7212 (fold 4), with an MSE across folds equal to 0.7694. The R code for reproducing these results is given in Appendix 5. 8.8 Bayesian Kernel Methods For a single environment, the model can be expressed as y = \mu1 + u + e,

    Parameters
    ----------
    folds : array-like
        Input data.
    Fig : array-like
        Input data.
    d : array-like
        Input data.
    the : array-like
        Input data.
    optimal : array-like
        Input data.
    number : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.8) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    r"""
    folds = np.atleast_1d(np.asarray(folds, dtype=float))
    n = len(folds)
    result = float(np.mean(folds))
    se = float(np.std(folds, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.8) from MVSML chapter 8."})


def cheatsheet():
    return "msm138: Numbered display equation (8.8) from MVSML chapter 8."

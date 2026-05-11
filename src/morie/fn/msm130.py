"""Numbered display equation (8.3) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_3"]


def mvsml_categorical_count_eq_8_3(needs, to, be, symmetric, positive, semi):
    """
    Numbered display equation (8.3) from MVSML chapter 8.

    Formula: needs to be symmetric and positive semi-deﬁnite, the term \betaTK\beta is an empirical RKHS norm with regard to the training data, \lambda is a smoothing or regularization parameter that should be positive and should control the trade-off between model goodness of ﬁt and complexity, and the factor 1 2 is introduced for convenience. The second term of (8.3) acts as a penalization term that is added to the minus log-likelihood. The goal is to ﬁnd \eta0 and \beta, which is equivalent to ﬁnding f xi ( ) = \eta0 + kT i \beta that minimizes

    Parameters
    ----------
    needs : array-like
        Input data.
    to : array-like
        Input data.
    be : array-like
        Input data.
    symmetric : array-like
        Input data.
    positive : array-like
        Input data.
    semi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.3) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    needs = np.atleast_1d(np.asarray(needs, dtype=float))
    n = len(needs)
    result = float(np.mean(needs))
    se = float(np.std(needs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.3) from MVSML chapter 8."})


def cheatsheet():
    return "msm130: Numbered display equation (8.3) from MVSML chapter 8."

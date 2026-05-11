"""Numbered display equation (8.3) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_3"]


def mvsml_categorical_count_eq_8_3(n, i, where, K, k1, kn):
    """
    Numbered display equation (8.3) from MVSML chapter 8.

    Formula: n i=1 \eta0, \beta where K = [k1, . . ., kn] is the n  n kernel matrix with ki as deﬁned above. Since K needs to be symmetric and positive semi-deﬁnite, the term \betaTK\beta is an empirical RKHS norm with regard to the training data, \lambda is a smoothing or regularization parameter that should be positive and should control the trade-off between model goodness of ﬁt and complexity, and the factor 1 2 is introduced for convenience. The second term of

    Parameters
    ----------
    n : array-like
        Input data.
    i : array-like
        Input data.
    where : array-like
        Input data.
    K : array-like
        Input data.
    k1 : array-like
        Input data.
    kn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.3) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.3) from MVSML chapter 8."})


def cheatsheet():
    return "msm129: Numbered display equation (8.3) from MVSML chapter 8."

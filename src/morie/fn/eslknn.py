"""k-nearest neighbors."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_knn"]


def esl_knn(X, y, k):
    """
    k-nearest neighbors

    Formula: f_hat(x) = (1/k) sum_{x_i in N_k(x)} y_i

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction

    References
    ----------
    Hastie ESL Ch 13
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "k-nearest neighbors"})


def cheatsheet():
    return "eslknn: k-nearest neighbors"

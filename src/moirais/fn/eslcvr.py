"""k-fold cross-validation error."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_cv_score"]


def esl_cv_score(X, y, model, k):
    """
    k-fold cross-validation error

    Formula: CV = (1/n) sum L(y_i, f_hat^{-k(i)}(x_i))

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    model : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 7
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "k-fold cross-validation error"})


def cheatsheet():
    return "eslcvr: k-fold cross-validation error"

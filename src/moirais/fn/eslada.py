"""AdaBoost.M1 weight update."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_adaboost"]


def esl_adaboost(X, y, M):
    """
    AdaBoost.M1 weight update

    Formula: alpha_m = log((1-err_m)/err_m)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Hastie ESL Ch 10
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AdaBoost.M1 weight update"})


def cheatsheet():
    return "eslada: AdaBoost.M1 weight update"

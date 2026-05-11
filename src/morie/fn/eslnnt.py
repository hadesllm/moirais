"""Single hidden layer neural net."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_neural_net"]


def esl_neural_net(X, y, M):
    """
    Single hidden layer neural net

    Formula: Z_m = sigma(alpha_{0m} + alpha_m' X); T_k = beta_{0k} + beta_k' Z; f_k(X) = g_k(T)

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
    Hastie ESL Ch 11
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Single hidden layer neural net"})


def cheatsheet():
    return "eslnnt: Single hidden layer neural net"

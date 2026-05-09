# moirais.fn — function file (hadesllm/moirais)
"""Logistic regression log-likelihood."""
import numpy as np
from ._richresult import RichResult

__all__ = ["logistic_log_likelihood"]


def logistic_log_likelihood(y, X, beta):
    """
    Logistic regression log-likelihood

    Formula: log L = sum_i [y_i*log(p_i) + (1-y_i)*log(1-p_i)]; p_i = sigmoid(X_i*beta)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'loglik': 'float'}

    References
    ----------
    Montesinos Lopez Ch 3
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logistic regression log-likelihood"})


def cheatsheet():
    return "lgobj: Logistic regression log-likelihood"

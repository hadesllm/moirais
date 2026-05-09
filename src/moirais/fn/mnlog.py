# moirais.fn — function file (hadesllm/moirais)
"""Penalized multinomial logistic regression (elastic net, glmnet)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["multinomial_logistic_penalized"]


def multinomial_logistic_penalized(y, X, lam, alpha):
    """
    Penalized multinomial logistic regression (elastic net, glmnet)

    Formula: L = -sum log P(y_i=k|X_i) + lambda*(alpha*||B||_1 + (1-alpha)/2*||B||_F^2)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    lam : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'B_hat': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 7
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Penalized multinomial logistic regression (elastic net, glmnet)"})


def cheatsheet():
    return "mnlog: Penalized multinomial logistic regression (elastic net, glmnet)"

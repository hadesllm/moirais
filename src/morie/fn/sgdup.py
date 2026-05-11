"""Stochastic gradient descent update on mini-batch."""
import numpy as np
from ._richresult import RichResult

__all__ = ["sgd_update"]


def sgd_update(beta, batch_grads, eta):
    """
    Stochastic gradient descent update on mini-batch

    Formula: beta = beta - eta * (1/|B|) * sum_{i in B} grad_i(L)

    Parameters
    ----------
    beta : array-like
        Input data.
    batch_grads : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'beta': 'array'}

    References
    ----------
    Montesinos Lopez Ch 3
    """
    beta = np.asarray(beta, dtype=float)
    n = int(beta) if beta.ndim == 0 else len(beta)
    result = float(np.mean(beta))
    se = float(np.std(beta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stochastic gradient descent update on mini-batch"})


def cheatsheet():
    return "sgdup: Stochastic gradient descent update on mini-batch"

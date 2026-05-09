# moirais.fn — function file (hadesllm/moirais)
"""Poisson log-likelihood loss for DNN count genomic outcomes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["poisson_loss_dnn"]


def poisson_loss_dnn(y, y_hat):
    """
    Poisson log-likelihood loss for DNN count genomic outcomes

    Formula: L = (1/n) * sum_i (exp(y_hat_i) - y_i * y_hat_i)

    Parameters
    ----------
    y : array-like
        Input data.
    y_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'loss': 'float'}

    References
    ----------
    Montesinos Lopez Ch 12
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Poisson log-likelihood loss for DNN count genomic outcomes"})


def cheatsheet():
    return "poilO: Poisson log-likelihood loss for DNN count genomic outcomes"

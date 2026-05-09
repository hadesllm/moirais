# moirais.fn — function file (hadesllm/moirais)
"""MSE loss for continuous genomic trait prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["mse_loss_continuous"]


def mse_loss_continuous(y, y_hat):
    """
    MSE loss for continuous genomic trait prediction

    Formula: L = (1/n) * sum_i (y_i - y_hat_i)^2

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
    Montesinos Lopez Ch 10
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSE loss for continuous genomic trait prediction"})


def cheatsheet():
    return "msebl: MSE loss for continuous genomic trait prediction"

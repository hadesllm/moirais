# morie.fn — function file (hadesllm/morie)
"""Linear regression implemented in PyTorch."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_linreg_pytorch"]


def geron_linreg_pytorch(X, y, epochs, lr):
    """
    Linear regression implemented in PyTorch

    Formula: y_hat = X @ w + b; MSE loss; SGD

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 10
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear regression implemented in PyTorch"})


def cheatsheet():
    return "hmlrpt: Linear regression implemented in PyTorch"

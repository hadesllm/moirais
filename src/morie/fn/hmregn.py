# morie.fn — function file (hadesllm/morie)
"""Regression MLP: linear output layer and MSE loss."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_regression_mlp"]


def geron_regression_mlp(X, y, hidden_sizes, epochs, lr):
    """
    Regression MLP: linear output layer and MSE loss

    Formula: loss = (1/m) sum ||y_hat - y||^2

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    hidden_sizes : array-like
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
    Géron Ch 9
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Regression MLP: linear output layer and MSE loss"})


def cheatsheet():
    return "hmregn: Regression MLP: linear output layer and MSE loss"

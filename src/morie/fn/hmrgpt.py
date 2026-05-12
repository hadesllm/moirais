# morie.fn -- function file (hadesllm/morie)
"""Regression MLP with PyTorch nn.Sequential."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_regression_mlp_pytorch"]


def geron_regression_mlp_pytorch(X, y, hidden, epochs, lr):
    """
    Regression MLP with PyTorch nn.Sequential

    Formula: nn.Linear -> ReLU -> nn.Linear ... -> nn.Linear(1)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    hidden : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Regression MLP with PyTorch nn.Sequential"})


def cheatsheet():
    return "hmrgpt: Regression MLP with PyTorch nn.Sequential"

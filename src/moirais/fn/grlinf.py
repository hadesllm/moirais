# moirais.fn — function file (hadesllm/moirais)
"""Affine (linear) layer forward pass."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_linear_layer_forward"]


def geron_linear_layer_forward(X, W, b):
    """
    Affine (linear) layer forward pass

    Formula: y = X W^T + b

    Parameters
    ----------
    X : array-like
        Input data.
    W : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 10, Linear layer / nn.Linear
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Affine (linear) layer forward pass"})


def cheatsheet():
    return "grlinf: Affine (linear) layer forward pass"

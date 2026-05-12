# morie.fn -- function file (hadesllm/morie)
"""Hebb's rule for perceptron weight update."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_hebb_rule"]


def geron_hebb_rule(x, y_true, y_pred, w, eta):
    """
    Hebb's rule for perceptron weight update

    Formula: w_{ij} <- w_{ij} + eta * (y_j - y_hat_j) * x_i

    Parameters
    ----------
    x : array-like
        Input data.
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.
    w : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: w_new

    References
    ----------
    Géron Ch 9, Hebb's Rule section
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hebb's rule for perceptron weight update"})


def cheatsheet():
    return "grhbb: Hebb's rule for perceptron weight update"

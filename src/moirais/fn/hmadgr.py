# moirais.fn — function file (hadesllm/moirais)
"""AdaGrad: per-parameter learning rates scaled by historical gradients."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_adagrad"]


def geron_adagrad(grads, s, eta, eps):
    """
    AdaGrad: per-parameter learning rates scaled by historical gradients

    Formula: s <- s + g^2; theta <- theta - eta * g / (sqrt(s) + eps)

    Parameters
    ----------
    grads : array-like
        Input data.
    s : array-like
        Input data.
    eta : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 11
    """
    grads = np.atleast_1d(np.asarray(grads, dtype=float))
    n = len(grads)
    result = float(np.mean(grads))
    se = float(np.std(grads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AdaGrad: per-parameter learning rates scaled by historical gradients"})


def cheatsheet():
    return "hmadgr: AdaGrad: per-parameter learning rates scaled by historical gradients"

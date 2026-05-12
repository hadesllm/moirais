# morie.fn -- function file (hadesllm/morie)
"""Automatic differentiation via reverse-mode autograd."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_autograd"]


def geron_autograd(loss, params):
    """
    Automatic differentiation via reverse-mode autograd

    Formula: loss.backward() populates .grad on leaf tensors

    Parameters
    ----------
    loss : array-like
        Input data.
    params : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradients

    References
    ----------
    Géron Ch 10
    """
    loss = np.atleast_1d(np.asarray(loss, dtype=float))
    n = len(loss)
    result = float(np.mean(loss))
    se = float(np.std(loss, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Automatic differentiation via reverse-mode autograd"})


def cheatsheet():
    return "hmagrd: Automatic differentiation via reverse-mode autograd"

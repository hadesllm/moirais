# morie.fn — function file (hadesllm/morie)
"""Dropout regularization: expected output under Bernoulli masking."""
import numpy as np
from ._richresult import RichResult

__all__ = ["dropout_regularization"]


def dropout_regularization(x, p):
    """
    Dropout regularization: expected output under Bernoulli masking

    Formula: E[f(x)] = f(x) * (1-p); mask each unit with prob p during training

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'masked_x': 'array'}

    References
    ----------
    Montesinos Lopez Ch 11
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dropout regularization: expected output under Bernoulli masking"})


def cheatsheet():
    return "dropr: Dropout regularization: expected output under Bernoulli masking"

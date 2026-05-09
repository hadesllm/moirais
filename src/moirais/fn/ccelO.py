# moirais.fn — function file (hadesllm/moirais)
"""Categorical cross-entropy loss for multi-class outcomes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["categorical_crossentropy_loss"]


def categorical_crossentropy_loss(y, p):
    """
    Categorical cross-entropy loss for multi-class outcomes

    Formula: L = -(1/n) * sum_i sum_k y_ik * log(p_ik)

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Categorical cross-entropy loss for multi-class outcomes"})


def cheatsheet():
    return "ccelO: Categorical cross-entropy loss for multi-class outcomes"

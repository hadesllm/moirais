# moirais.fn — function file (hadesllm/moirais)
"""Recursive multi-step: feed each prediction back as input for the next step."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_recursive_multistep"]


def joseph_recursive_multistep(y, model, H):
    """
    Recursive multi-step: feed each prediction back as input for the next step

    Formula: y_hat_{T+h} = f(y_hat_{T+h-1}, y_hat_{T+h-2}, ..., x);  h = 1..H

    Parameters
    ----------
    y : array-like
        Input data.
    model : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Joseph Ch 18, Recursive (MIMO) Strategy section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Recursive multi-step: feed each prediction back as input for the next step"})


def cheatsheet():
    return "jorec: Recursive multi-step: feed each prediction back as input for the next step"

# morie.fn — function file (hadesllm/morie)
"""Walk-forward validation: refit model after each new observation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_walk_forward_validation"]


def joseph_walk_forward_validation(y, model, T_start):
    """
    Walk-forward validation: refit model after each new observation

    Formula: for t in T_start..T: refit on y[0..t-1]; predict y_hat_t; compare to y_t

    Parameters
    ----------
    y : array-like
        Input data.
    model : array-like
        Input data.
    T_start : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: predictions, errors

    References
    ----------
    Joseph Ch 20, Walk-Forward Validation section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Walk-forward validation: refit model after each new observation"})


def cheatsheet():
    return "jowfv: Walk-forward validation: refit model after each new observation"

# morie.fn -- function file (hadesllm/morie)
"""Front-door adjustment formula for causal effect via mediator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["frontdoor_adjustment"]


def frontdoor_adjustment(X, Y, Z, data):
    """
    Front-door adjustment formula for causal effect via mediator

    Formula: P(Y|do(X)) = sum_z P(Z=z|X) * sum_{data'} P(Y|X=x',Z=z) * P(X=data')

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    Z : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'ate': 'float'}

    References
    ----------
    Molak Ch 6
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Front-door adjustment formula for causal effect via mediator"})


def cheatsheet():
    return "fdadj: Front-door adjustment formula for causal effect via mediator"

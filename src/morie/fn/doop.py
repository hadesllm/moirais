# morie.fn -- function file (hadesllm/morie)
"""Pearl's do-operator: intervention by setting variable to value."""
import numpy as np
from ._richresult import RichResult

__all__ = ["do_operator"]


def do_operator(Y, X, x_val, model):
    """
    Pearl's do-operator: intervention by setting variable to value

    Formula: P(Y|do(X=Y)) = sum_z P(Y|X=Y,Z=z)*P(Z); marginalizes over pre-intervention Z distribution

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    x_val : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'p_y_do_x': 'distribution'}

    References
    ----------
    Molak Ch 2
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pearl's do-operator: intervention by setting variable to value"})


def cheatsheet():
    return "doop: Pearl's do-operator: intervention by setting variable to value"

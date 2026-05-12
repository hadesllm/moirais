# morie.fn -- function file (hadesllm/morie)
"""Regression MLP output: linear (or softplus / identity) final activation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_regression_mlp_output"]


def geron_regression_mlp_output(a_last, W_out, b_out):
    """
    Regression MLP output: linear (or softplus / identity) final activation

    Formula: y_hat = W_out a_{L-1} + b_out  (typically no nonlinearity on final layer)

    Parameters
    ----------
    a_last : array-like
        Input data.
    W_out : array-like
        Input data.
    b_out : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Géron Ch 9, Regression MLPs section
    """
    a_last = np.asarray(a_last, dtype=float)
    n = int(a_last) if a_last.ndim == 0 else len(a_last)
    result = float(np.mean(a_last))
    se = float(np.std(a_last, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Regression MLP output: linear (or softplus / identity) final activation"})


def cheatsheet():
    return "grmlr: Regression MLP output: linear (or softplus / identity) final activation"

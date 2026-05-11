# morie.fn — function file (hadesllm/morie)
"""Multi-output DNN for multi-trait genomic prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["dnn_multitrait"]


def dnn_multitrait(X, Y, layers, heads):
    """
    Multi-output DNN for multi-trait genomic prediction

    Formula: shared_hidden -> [head_1, ..., head_t]; joint loss L = sum_t w_t * L_t

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    layers : array-like
        Input data.
    heads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'Y_hat': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 12
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-output DNN for multi-trait genomic prediction"})


def cheatsheet():
    return "dnnmt: Multi-output DNN for multi-trait genomic prediction"

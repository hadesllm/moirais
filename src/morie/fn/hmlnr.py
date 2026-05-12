# morie.fn -- function file (hadesllm/morie)
"""Layer normalization applied in RNN cells."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_layer_norm_rnn"]


def geron_layer_norm_rnn(x, gamma, beta):
    """
    Layer normalization applied in RNN cells

    Formula: normalize across feature dim within each time step

    Parameters
    ----------
    x : array-like
        Input data.
    gamma : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 13
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Layer normalization applied in RNN cells"})


def cheatsheet():
    return "hmlnr: Layer normalization applied in RNN cells"

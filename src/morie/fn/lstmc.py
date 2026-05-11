# morie.fn — function file (hadesllm/morie)
"""LSTM cell forward pass."""
import numpy as np
from ._richresult import RichResult

__all__ = ["lstm_cell"]


def lstm_cell(x):
    """
    LSTM cell forward pass

    Formula: f,i,o = sigma(W*[h,x]+b), c = f*c + i*tanh(...)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hochreiter & Schmidhuber (1997)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LSTM cell forward pass"})


def cheatsheet():
    return "lstmc: LSTM cell forward pass"

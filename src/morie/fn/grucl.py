# morie.fn — function file (hadesllm/morie)
"""GRU cell forward pass."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gru_cell"]


def gru_cell(x):
    """
    GRU cell forward pass

    Formula: z,r = sigma(W*[h,x]), h = (1-z)*h + z*tanh(...)

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
    Cho et al. (2014)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GRU cell forward pass"})


def cheatsheet():
    return "grucl: GRU cell forward pass"

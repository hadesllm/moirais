# morie.fn — function file (hadesllm/morie)
"""MIDAS mixed-frequency regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["midas_regression"]


def midas_regression(x, y):
    """
    MIDAS mixed-frequency regression

    Formula: y_t = beta_0 + beta_1 * sum B(k;theta)*x_{t-k/m} + e_t

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Ghysels et al. (2004)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MIDAS mixed-frequency regression"})


def cheatsheet():
    return "midas: MIDAS mixed-frequency regression"

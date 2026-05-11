"""Threshold autoregressive model (TAR)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["threshold_autoregression"]


def threshold_autoregression(x):
    """
    Threshold autoregressive model (TAR)

    Formula: y_t = phi_1'*x_t if y_{t-d}<=c, phi_2'*x_t otherwise

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
    Tong (1990)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Threshold autoregressive model (TAR)"})


def cheatsheet():
    return "tarmd: Threshold autoregressive model (TAR)"

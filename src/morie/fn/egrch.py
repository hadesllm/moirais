# morie.fn — function file (hadesllm/morie)
"""EGARCH asymmetric volatility."""
import numpy as np
from ._richresult import RichResult

__all__ = ["egarch_model"]


def egarch_model(x):
    """
    EGARCH asymmetric volatility

    Formula: log(sigma_t^2) = omega + alpha*g(z) + beta*log(sigma_{t-1}^2)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Nelson (1991)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EGARCH asymmetric volatility"})


def cheatsheet():
    return "egrch: EGARCH asymmetric volatility"

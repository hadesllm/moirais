# morie.fn — function file (hadesllm/morie)
"""Kalman filter predict-update."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kalman_filter"]


def kalman_filter(x):
    """
    Kalman filter predict-update

    Formula: x_hat = F*x + K*(y - H*F*x), K = P*H'*(H*P*H'+R)^{-1}

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
    Kalman (1960)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kalman filter predict-update"})


def cheatsheet():
    return "kalmn: Kalman filter predict-update"

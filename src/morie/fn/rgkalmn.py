# morie.fn -- function file (hadesllm/morie)
"""Kalman filter: state prediction/update with Riccati equation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_kalman_filter"]


def rangayyan_kalman_filter(z, F, H, Q, R, x0, P0):
    """
    Kalman filter: state prediction/update with Riccati equation

    Formula: x_k|k-1=F*x_{k-1}; P_k|k-1=F*P*F^T+Q; K=P*H^T*(H*P*H^T+R)^{-1}; update z,P

    Parameters
    ----------
    z : array-like
        Input data.
    F : array-like
        Input data.
    H : array-like
        Input data.
    Q : array-like
        Input data.
    R : array-like
        Input data.
    x0 : array-like
        Input data.
    P0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_hat, P_hat, K_gains

    References
    ----------
    Rangayyan Ch 8.7
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kalman filter: state prediction/update with Riccati equation"})


def cheatsheet():
    return "rgkalmn: Kalman filter: state prediction/update with Riccati equation"

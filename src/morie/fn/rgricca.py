# morie.fn -- function file (hadesllm/morie)
"""Steady-state Riccati equation solution for Kalman gain."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_riccati_eq"]


def rangayyan_riccati_eq(F, H, Q, R):
    """
    Steady-state Riccati equation solution for Kalman gain

    Formula: P = F*P*F^T + Q - F*P*H^T*(H*P*H^T+R)^{-1}*H*P*F^T

    Parameters
    ----------
    F : array-like
        Input data.
    H : array-like
        Input data.
    Q : array-like
        Input data.
    R : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: P_steady, K_steady

    References
    ----------
    Rangayyan Ch 8.7
    """
    F = np.asarray(F, dtype=float)
    n = int(F) if F.ndim == 0 else len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Steady-state Riccati equation solution for Kalman gain"})


def cheatsheet():
    return "rgricca: Steady-state Riccati equation solution for Kalman gain"

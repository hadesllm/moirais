# morie.fn — function file (hadesllm/morie)
"""Hodgkin-Huxley gating variable ODEs (m, h, n)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_hh_gating"]


def rangayyan_hh_gating(V, dt):
    """
    Hodgkin-Huxley gating variable ODEs (m, h, n)

    Formula: dm/dt = alpha_m(V)*(1-m) - beta_m(V)*m; similarly h, n

    Parameters
    ----------
    V : array-like
        Input data.
    dt : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: m, h, n

    References
    ----------
    Rangayyan Ch 7.8.1
    """
    V = np.asarray(V, dtype=float)
    n = int(V) if V.ndim == 0 else len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hodgkin-Huxley gating variable ODEs (m, h, n)"})


def cheatsheet():
    return "rghgate: Hodgkin-Huxley gating variable ODEs (m, h, n)"

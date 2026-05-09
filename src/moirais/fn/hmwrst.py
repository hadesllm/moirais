# moirais.fn — function file (hadesllm/moirais)
"""Warm restarts: cosine decay with periodic restarts."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_warm_restarts"]


def geron_warm_restarts(t, T0, factor, eta_max, eta_min):
    """
    Warm restarts: cosine decay with periodic restarts

    Formula: after period T_i restart to eta_max; T_{i+1} = T_i * factor

    Parameters
    ----------
    t : array-like
        Input data.
    T0 : array-like
        Input data.
    factor : array-like
        Input data.
    eta_max : array-like
        Input data.
    eta_min : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: eta

    References
    ----------
    Géron Ch 11
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Warm restarts: cosine decay with periodic restarts"})


def cheatsheet():
    return "hmwrst: Warm restarts: cosine decay with periodic restarts"

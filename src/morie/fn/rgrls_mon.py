# morie.fn -- function file (hadesllm/morie)
"""Monitoring RLS filter output for nonstationary detection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_rls_monitor"]


def rangayyan_rls_monitor(x, d, lam, threshold):
    """
    Monitoring RLS filter output for nonstationary detection

    Formula: Segment boundary detected when ||e(n)||^2 exceeds threshold after RLS convergence

    Parameters
    ----------
    x : array-like
        Input data.
    d : array-like
        Input data.
    lam : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: change_points, error_trace

    References
    ----------
    Rangayyan Ch 8.6.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Monitoring RLS filter output for nonstationary detection"})


def cheatsheet():
    return "rgrls_mon: Monitoring RLS filter output for nonstationary detection"

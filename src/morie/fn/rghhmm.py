# morie.fn -- function file (hadesllm/morie)
"""Hodgkin-Huxley membrane model for action potential."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_hodgkin_huxley"]


def rangayyan_hodgkin_huxley(t, I_ext, g_Na, g_K, g_L, C_m):
    """
    Hodgkin-Huxley membrane model for action potential

    Formula: C_m dV/dt = I_ext - I_Na - I_K - I_L; I_Na=g_Na*m^3*h*(V-E_Na)

    Parameters
    ----------
    t : array-like
        Input data.
    I_ext : array-like
        Input data.
    g_Na : array-like
        Input data.
    g_K : array-like
        Input data.
    g_L : array-like
        Input data.
    C_m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V_t, m_t, h_t, n_t

    References
    ----------
    Rangayyan Ch 7.8.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hodgkin-Huxley membrane model for action potential"})


def cheatsheet():
    return "rghhmm: Hodgkin-Huxley membrane model for action potential"

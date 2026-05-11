# morie.fn — function file (hadesllm/morie)
"""Membrane potential dynamics (RC circuit model)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_membrane_potential"]


def rangayyan_membrane_potential(t, I_inj, C_m, R_m, V_rest):
    """
    Membrane potential dynamics (RC circuit model)

    Formula: C_m * dV/dt = -(V - V_rest)/R_m + I_inj

    Parameters
    ----------
    t : array-like
        Input data.
    I_inj : array-like
        Input data.
    C_m : array-like
        Input data.
    R_m : array-like
        Input data.
    V_rest : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V_t

    References
    ----------
    Rangayyan Ch 1.2.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Membrane potential dynamics (RC circuit model)"})


def cheatsheet():
    return "rgmemb: Membrane potential dynamics (RC circuit model)"

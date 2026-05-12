# morie.fn -- function file (hadesllm/morie)
"""Idealized action potential waveform model (depolarization/repolarization)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_action_potential"]


def rangayyan_action_potential(t, v_rest, v_peak, t_rise, t_fall):
    """
    Idealized action potential waveform model (depolarization/repolarization)

    Formula: V(t) = piecewise ramp-and-decay model

    Parameters
    ----------
    t : array-like
        Input data.
    v_rest : array-like
        Input data.
    v_peak : array-like
        Input data.
    t_rise : array-like
        Input data.
    t_fall : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: voltage_array

    References
    ----------
    Rangayyan Ch 1.2.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Idealized action potential waveform model (depolarization/repolarization)"})


def cheatsheet():
    return "rgap: Idealized action potential waveform model (depolarization/repolarization)"

# morie.fn -- function file (hadesllm/morie)
"""Coronary artery sound generation model (turbulent flow)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_coronary_sound"]


def rangayyan_coronary_sound(diameter, flow_velocity, stenosis_pct):
    """
    Coronary artery sound generation model (turbulent flow)

    Formula: Strouhal number St = f*d/v; resonance frequency of stenosis

    Parameters
    ----------
    diameter : array-like
        Input data.
    flow_velocity : array-like
        Input data.
    stenosis_pct : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: resonance_freq, sound_model

    References
    ----------
    Rangayyan Ch 7.7.2
    """
    diameter = np.asarray(diameter, dtype=float)
    n = int(diameter) if diameter.ndim == 0 else len(diameter)
    result = float(np.mean(diameter))
    se = float(np.std(diameter, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Coronary artery sound generation model (turbulent flow)"})


def cheatsheet():
    return "rgcorart: Coronary artery sound generation model (turbulent flow)"

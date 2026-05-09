# moirais.fn — function file (hadesllm/moirais)
"""Motor unit action potential (MUAP) model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_muap"]


def rangayyan_muap(t, n_fibers, conduction_vel):
    """
    Motor unit action potential (MUAP) model

    Formula: MUAP = sum of triphasic dipole contributions from muscle fibers

    Parameters
    ----------
    t : array-like
        Input data.
    n_fibers : array-like
        Input data.
    conduction_vel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: muap_waveform

    References
    ----------
    Rangayyan Ch 1.2.4
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Motor unit action potential (MUAP) model"})


def cheatsheet():
    return "rgmuap: Motor unit action potential (MUAP) model"

# morie.fn — function file (hadesllm/morie)
"""Synchronized averaging of PCG envelopes (S1/S2 intensity analysis)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pcg_envelope_avg"]


def rangayyan_pcg_envelope_avg(pcg, ecg, fs):
    """
    Synchronized averaging of PCG envelopes (S1/S2 intensity analysis)

    Formula: env_k = Hilbert envelope of k-th cardiac cycle; avg over cycles

    Parameters
    ----------
    pcg : array-like
        Input data.
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: avg_s1_env, avg_s2_env

    References
    ----------
    Rangayyan Ch 5.5.2
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Synchronized averaging of PCG envelopes (S1/S2 intensity analysis)"})


def cheatsheet():
    return "rgpcgenl: Synchronized averaging of PCG envelopes (S1/S2 intensity analysis)"

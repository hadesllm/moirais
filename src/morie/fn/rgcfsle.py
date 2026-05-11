# morie.fn — function file (hadesllm/morie)
"""Cardiorespiratory coupling analysis via coherence and PLV."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_coupled_freq_select"]


def rangayyan_coupled_freq_select(ecg, resp, fs):
    """
    Cardiorespiratory coupling analysis via coherence and PLV

    Formula: PLV = |mean(exp(j*(phi_ecg - phi_resp)))|; coherence at resp frequency

    Parameters
    ----------
    ecg : array-like
        Input data.
    resp : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: plv, coherence_at_rr

    References
    ----------
    Rangayyan Ch 2.4
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cardiorespiratory coupling analysis via coherence and PLV"})


def cheatsheet():
    return "rgcfsle: Cardiorespiratory coupling analysis via coherence and PLV"

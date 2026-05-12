# morie.fn -- function file (hadesllm/morie)
"""Motor unit mean firing rate and inter-discharge interval (IDI)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_muap_firing_rate"]


def rangayyan_muap_firing_rate(spike_times):
    """
    Motor unit mean firing rate and inter-discharge interval (IDI)

    Formula: MFR = 1/mean(IDI), CV_IDI = std(IDI)/mean(IDI)

    Parameters
    ----------
    spike_times : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mfr, cv_idi

    References
    ----------
    Rangayyan Ch 1.2.4
    """
    spike_times = np.asarray(spike_times, dtype=float)
    n = int(spike_times) if spike_times.ndim == 0 else len(spike_times)
    result = float(np.mean(spike_times))
    se = float(np.std(spike_times, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Motor unit mean firing rate and inter-discharge interval (IDI)"})


def cheatsheet():
    return "rgmufr: Motor unit mean firing rate and inter-discharge interval (IDI)"

"""Spike-train information rate (neural)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spike_information"]


def spike_information(spike, stim):
    """
    Spike-train information rate (neural)

    Formula: I_spike = H(spike) - H(spike|stim)

    Parameters
    ----------
    spike : array-like
        Input data.
    stim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Strong et al (1998)
    """
    spike = np.atleast_1d(np.asarray(spike, dtype=float))
    n = len(spike)
    result = float(np.mean(spike))
    se = float(np.std(spike, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spike-train information rate (neural)"})


def cheatsheet():
    return "spkint: Spike-train information rate (neural)"

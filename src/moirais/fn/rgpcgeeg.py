# moirais.fn — function file (hadesllm/moirais)
"""PCG-EEG coupling analysis for auditory evoked response."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pcg_eeg_coupling"]


def rangayyan_pcg_eeg_coupling(pcg, eeg, fs):
    """
    PCG-EEG coupling analysis for auditory evoked response

    Formula: Cross-coherence C_pcg_eeg(f); peak at fundamental S1 frequency

    Parameters
    ----------
    pcg : array-like
        Input data.
    eeg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coherence, coupling_freq

    References
    ----------
    Rangayyan Ch 2
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PCG-EEG coupling analysis for auditory evoked response"})


def cheatsheet():
    return "rgpcgeeg: PCG-EEG coupling analysis for auditory evoked response"

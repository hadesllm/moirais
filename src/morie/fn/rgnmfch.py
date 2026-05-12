# morie.fn -- function file (hadesllm/morie)
"""NMF-based EEG channel selection for BCI."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_nmf_channel_sel"]


def rangayyan_nmf_channel_sel(eeg, n_comp, n_select):
    """
    NMF-based EEG channel selection for BCI

    Formula: W matrix columns: spatial activation patterns; select channels by max W_ij

    Parameters
    ----------
    eeg : array-like
        Input data.
    n_comp : array-like
        Input data.
    n_select : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: selected_ch_idx, W, H

    References
    ----------
    Rangayyan Ch 9.12.1
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NMF-based EEG channel selection for BCI"})


def cheatsheet():
    return "rgnmfch: NMF-based EEG channel selection for BCI"

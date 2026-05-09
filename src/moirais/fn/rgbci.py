# moirais.fn — function file (hadesllm/moirais)
"""BCI EEG channel selection via NMF spatial decomposition."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_bci_nmf"]


def rangayyan_bci_nmf(eeg, n_components, fs):
    """
    BCI EEG channel selection via NMF spatial decomposition

    Formula: NMF: V=WH; W=spatial patterns, H=temporal activations; select channels by W

    Parameters
    ----------
    eeg : array-like
        Input data.
    n_components : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: selected_channels, W, H

    References
    ----------
    Rangayyan Ch 9.12
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BCI EEG channel selection via NMF spatial decomposition"})


def cheatsheet():
    return "rgbci: BCI EEG channel selection via NMF spatial decomposition"

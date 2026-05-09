# moirais.fn — function file (hadesllm/moirais)
"""Synchronized (ensemble) averaging for SNR enhancement."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_sync_average"]


def rangayyan_sync_average(epochs):
    """
    Synchronized (ensemble) averaging for SNR enhancement

    Formula: x_avg[n] = (1/M) sum_{k=1}^{M} x_k[n]; SNR = sqrt(M) * signal_SNR

    Parameters
    ----------
    epochs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: averaged_signal, snr_gain

    References
    ----------
    Rangayyan Ch 3.5
    """
    epochs = np.asarray(epochs, dtype=float)
    n = int(epochs) if epochs.ndim == 0 else len(epochs)
    result = float(np.mean(epochs))
    se = float(np.std(epochs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Synchronized (ensemble) averaging for SNR enhancement"})


def cheatsheet():
    return "rgsavg: Synchronized (ensemble) averaging for SNR enhancement"

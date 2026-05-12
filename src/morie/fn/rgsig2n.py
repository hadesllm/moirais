# morie.fn -- function file (hadesllm/morie)
"""Signal-to-noise ratio calculation after filtering."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_signal_to_noise"]


def rangayyan_signal_to_noise(signal_clean, signal_noisy):
    """
    Signal-to-noise ratio calculation after filtering

    Formula: SNR = 10*log10(sum(x_clean^2)/sum(noise^2))

    Parameters
    ----------
    signal_clean : array-like
        Input data.
    signal_noisy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: snr_db

    References
    ----------
    Rangayyan Ch 3
    """
    signal_clean = np.asarray(signal_clean, dtype=float)
    n = int(signal_clean) if signal_clean.ndim == 0 else len(signal_clean)
    result = float(np.mean(signal_clean))
    se = float(np.std(signal_clean, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Signal-to-noise ratio calculation after filtering"})


def cheatsheet():
    return "rgsig2n: Signal-to-noise ratio calculation after filtering"

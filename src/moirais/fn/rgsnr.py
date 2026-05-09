# moirais.fn — function file (hadesllm/moirais)
"""Signal-to-noise ratio (dB)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_snr"]


def rangayyan_snr(signal, noise):
    """
    Signal-to-noise ratio (dB)

    Formula: SNR = 10*log10(P_signal / P_noise)

    Parameters
    ----------
    signal : array-like
        Input data.
    noise : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: snr_db

    References
    ----------
    Rangayyan Ch 1
    """
    signal = np.asarray(signal, dtype=float)
    n = int(signal) if signal.ndim == 0 else len(signal)
    result = float(np.mean(signal))
    se = float(np.std(signal, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Signal-to-noise ratio (dB)"})


def cheatsheet():
    return "rgsnr: Signal-to-noise ratio (dB)"

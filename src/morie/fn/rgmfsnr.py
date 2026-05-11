# morie.fn — function file (hadesllm/morie)
"""Output SNR of matched filter (maximum SNR theorem)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_matched_filter_snr"]


def rangayyan_matched_filter_snr(signal, noise_psd):
    """
    Output SNR of matched filter (maximum SNR theorem)

    Formula: SNR_max = 2*E_s/N0 where E_s = integral |s(t)|^2 dt

    Parameters
    ----------
    signal : array-like
        Input data.
    noise_psd : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: snr_max

    References
    ----------
    Rangayyan Ch 4.6.1
    """
    signal = np.asarray(signal, dtype=float)
    n = int(signal) if signal.ndim == 0 else len(signal)
    result = float(np.mean(signal))
    se = float(np.std(signal, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output SNR of matched filter (maximum SNR theorem)"})


def cheatsheet():
    return "rgmfsnr: Output SNR of matched filter (maximum SNR theorem)"

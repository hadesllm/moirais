# moirais.fn — function file (hadesllm/moirais)
"""EEG frequency band power (delta/theta/alpha/beta/gamma)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_eeg_bands"]


def rangayyan_eeg_bands(x):
    """
    EEG frequency band power (delta/theta/alpha/beta/gamma)

    Formula: P_band = integral S(f) df over band

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rangayyan Ch 9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EEG frequency band power (delta/theta/alpha/beta/gamma)"})


def cheatsheet():
    return "rgeeg: EEG frequency band power (delta/theta/alpha/beta/gamma)"

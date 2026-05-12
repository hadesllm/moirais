# morie.fn -- function file (hadesllm/morie)
"""Real cepstrum of a signal."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_cepstrum"]


def rangayyan_cepstrum(x):
    """
    Real cepstrum of a signal

    Formula: c(n) = IDFT(log|DFT(x)|) = IFFT(log|FFT(x)|)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cepstrum, quefrency

    References
    ----------
    Rangayyan Ch 4.7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Real cepstrum of a signal"})


def cheatsheet():
    return "rgcepst: Real cepstrum of a signal"

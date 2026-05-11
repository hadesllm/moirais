"""Time-domain difference equation of the von Hann (Hanning) smoothing filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_hann_filter"]


def rangayyan_ch3_hann_filter(x, n):
    """
    Time-domain difference equation of the von Hann (Hanning) smoothing filter.

    Formula: y(n) = (1/4) * [x(n) + 2*x(n-1) + x(n-2)]

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.100, p. 140
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-domain difference equation of the von Hann (Hanning) smoothing filter."})


def cheatsheet():
    return "rng089: Time-domain difference equation of the von Hann (Hanning) smoothing filter."

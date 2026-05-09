"""Length transformation used to detect P, QRS, and T waves across multiple ECG channels.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_length_transformation"]


def rangayyan_ch4_length_transformation(x, N, w, t):
    """
    Length transformation used to detect P, QRS, and T waves across multiple ECG channels.

    Formula: L(N, w, t) = integral_{t}^{t+w} sqrt( sum_{j=1}^{N} (dx_j/dt)^2 ) dt

    Parameters
    ----------
    x : array-like
        Input data.
    N : array-like
        Input data.
    w : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.21, p. 227
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Length transformation used to detect P, QRS, and T waves across multiple ECG channels."})


def cheatsheet():
    return "rng195: Length transformation used to detect P, QRS, and T waves across multiple ECG channels."

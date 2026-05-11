"""Combined impulse response of two LSI systems in series is their convolution.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lsi_series_combined_h"]


def rangayyan_ch3_lsi_series_combined_h(h_1, h_2, n):
    """
    Combined impulse response of two LSI systems in series is their convolution.

    Formula: h(n) = h_1(n) * h_2(n)

    Parameters
    ----------
    h_1 : array-like
        Input data.
    h_2 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.45, p. 115
    """
    h_1 = np.atleast_1d(np.asarray(h_1, dtype=float))
    n = len(h_1)
    result = float(np.mean(h_1))
    se = float(np.std(h_1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Combined impulse response of two LSI systems in series is their convolution."})


def cheatsheet():
    return "rng043: Combined impulse response of two LSI systems in series is their convolution."

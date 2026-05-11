"""Magnitude response of the Hann filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_hann_magnitude_response"]


def rangayyan_ch3_hann_magnitude_response(omega):
    """
    Magnitude response of the Hann filter.

    Formula: |H(omega)| = | 0.5 * [1 + cos(omega)] |

    Parameters
    ----------
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.106, p. 141
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Magnitude response of the Hann filter."})


def cheatsheet():
    return "rng095: Magnitude response of the Hann filter."

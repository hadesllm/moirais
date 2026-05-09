"""Linear phase response of the Hann filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_hann_phase_response"]


def rangayyan_ch3_hann_phase_response(omega):
    """
    Linear phase response of the Hann filter.

    Formula: angle(H(omega)) = -omega

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
    Rangayyan (2024), Ch 3, Eq 3.107, p. 141
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear phase response of the Hann filter."})


def cheatsheet():
    return "rng096: Linear phase response of the Hann filter."

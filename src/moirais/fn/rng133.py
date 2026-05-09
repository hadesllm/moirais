"""Direct discrete-domain specification of the Butterworth lowpass response.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_butterworth_lowpass_direct_specification"]


def rangayyan_ch3_butterworth_lowpass_direct_specification(omega, omega_c, N):
    """
    Direct discrete-domain specification of the Butterworth lowpass response.

    Formula: |H(omega)|^2 = 1 / (1 + (omega/omega_c)^(2*N))

    Parameters
    ----------
    omega : array-like
        Input data.
    omega_c : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.145, p. 155
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Direct discrete-domain specification of the Butterworth lowpass response."})


def cheatsheet():
    return "rng133: Direct discrete-domain specification of the Butterworth lowpass response."

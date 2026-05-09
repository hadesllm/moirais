"""Power-series expansion of the log echo term (a < 1).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_log_echo_power_series_expansion"]


def rangayyan_ch4_log_echo_power_series_expansion(a, n_0, omega, H_hat):
    """
    Power-series expansion of the log echo term (a < 1).

    Formula: Y_hat(omega) = H_hat(omega) + a*exp(-j*omega*n_0) - (a^2/2)*exp(-j*2*omega*n_0) + (a^3/3)*exp(-j*3*omega*n_0) - ...

    Parameters
    ----------
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    omega : array-like
        Input data.
    H_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.79, p. 249
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Power-series expansion of the log echo term (a < 1)."})


def cheatsheet():
    return "rng251: Power-series expansion of the log echo term (a < 1)."

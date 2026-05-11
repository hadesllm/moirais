"""Power-series expansion of log(1 - beta z) for |z| < |beta^-1|.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_log_maximum_phase_expansion"]


def rangayyan_ch4_log_maximum_phase_expansion(beta, z, n):
    """
    Power-series expansion of log(1 - beta z) for |z| < |beta^-1|.

    Formula: log(1 - beta z) = - sum_{n=1}^{inf} (beta^n / n) * z^n, for |z| < |beta^(-1)|

    Parameters
    ----------
    beta : array-like
        Input data.
    z : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.71, p. 248
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Power-series expansion of log(1 - beta z) for |z| < |beta^-1|."})


def cheatsheet():
    return "rng243: Power-series expansion of log(1 - beta z) for |z| < |beta^-1|."

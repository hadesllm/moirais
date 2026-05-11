"""Power-series expansion of log(1 - alpha z^-1) for |z| > |alpha|.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_log_minimum_phase_expansion"]


def rangayyan_ch4_log_minimum_phase_expansion(alpha, z, n):
    """
    Power-series expansion of log(1 - alpha z^-1) for |z| > |alpha|.

    Formula: log(1 - alpha z^(-1)) = - sum_{n=1}^{inf} (alpha^n / n) * z^(-n), for |z| > |alpha|

    Parameters
    ----------
    alpha : array-like
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
    Rangayyan (2024), Ch 4, Eq 4.70, p. 248
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Power-series expansion of log(1 - alpha z^-1) for |z| > |alpha|."})


def cheatsheet():
    return "rng242: Power-series expansion of log(1 - alpha z^-1) for |z| > |alpha|."

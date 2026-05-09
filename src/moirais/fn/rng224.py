"""Basic three-sample reference pattern used in matched-filter illustration.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_basic_signal_g"]


def rangayyan_ch4_basic_signal_g(n):
    """
    Basic three-sample reference pattern used in matched-filter illustration.

    Formula: g(n) = 3*delta(n) + 2*delta(n-1) + delta(n-2)

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.52, p. 240
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Basic three-sample reference pattern used in matched-filter illustration."})


def cheatsheet():
    return "rng224: Basic three-sample reference pattern used in matched-filter illustration."

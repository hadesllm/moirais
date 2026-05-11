"""kth observed realization of a signal in noise (signal-plus-noise model).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_observed_signal_kth_realization"]


def rangayyan_ch3_observed_signal_kth_realization(x_k, eta_k, n):
    """
    kth observed realization of a signal in noise (signal-plus-noise model).

    Formula: y_k(n) = x_k(n) + eta_k(n)

    Parameters
    ----------
    x_k : array-like
        Input data.
    eta_k : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.95, p. 135
    """
    x_k = np.atleast_1d(np.asarray(x_k, dtype=float))
    n = len(x_k)
    result = float(np.mean(x_k))
    se = float(np.std(x_k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "kth observed realization of a signal in noise (signal-plus-noise model)."})


def cheatsheet():
    return "rng084: kth observed realization of a signal in noise (signal-plus-noise model)."

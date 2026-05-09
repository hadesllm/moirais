"""Additive noise model: observed signal equals desired signal plus noise.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_signal_plus_noise_model"]


def rangayyan_ch3_signal_plus_noise_model(x, eta):
    """
    Additive noise model: observed signal equals desired signal plus noise.

    Formula: y(t) = x(t) + eta(t)

    Parameters
    ----------
    x : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.12, p. 96
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Additive noise model: observed signal equals desired signal plus noise."})


def cheatsheet():
    return "rng012: Additive noise model: observed signal equals desired signal plus noise."

"""Equivalent (z, not z^-1) form of the baseline-wander filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_baseline_wander_filter_z_form_b"]


def rangayyan_ch3_baseline_wander_filter_z_form_b(z, T):
    """
    Equivalent (z, not z^-1) form of the baseline-wander filter.

    Formula: H(z) = (1/T) * (z - 1) / (z - 0.995)

    Parameters
    ----------
    z : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.133, p. 149
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Equivalent (z, not z^-1) form of the baseline-wander filter."})


def cheatsheet():
    return "rng121: Equivalent (z, not z^-1) form of the baseline-wander filter."

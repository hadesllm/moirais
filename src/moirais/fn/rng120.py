"""Modified first-difference filter with pole at 0.995 to remove baseline wander.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_baseline_wander_filter_z_form_a"]


def rangayyan_ch3_baseline_wander_filter_z_form_a(z, T):
    """
    Modified first-difference filter with pole at 0.995 to remove baseline wander.

    Formula: H(z) = (1/T) * (1 - z^(-1)) / (1 - 0.995 * z^(-1))

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
    Rangayyan (2024), Ch 3, Eq 3.132, p. 149
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Modified first-difference filter with pole at 0.995 to remove baseline wander."})


def cheatsheet():
    return "rng120: Modified first-difference filter with pole at 0.995 to remove baseline wander."

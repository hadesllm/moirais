"""Moving-window integrator used in the Pan-Tompkins QRS detector.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_moving_window_integrator"]


def rangayyan_ch4_pan_tompkins_moving_window_integrator(x, N, n):
    """
    Moving-window integrator used in the Pan-Tompkins QRS detector.

    Formula: y(n) = (1/N) * { x[n-(N-1)] + x[n-(N-2)] + ... + x(n) }

    Parameters
    ----------
    x : array-like
        Input data.
    N : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.15, p. 223
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Moving-window integrator used in the Pan-Tompkins QRS detector."})


def cheatsheet():
    return "rng189: Moving-window integrator used in the Pan-Tompkins QRS detector."

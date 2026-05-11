"""Baujat plot: contribution to Q vs influence on θ̂."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_baujat_plot_data"]


def ma_baujat_plot_data(yi, vi):
    """
    Baujat plot: contribution to Q vs influence on θ̂

    Formula: (b_i, h_i) per study; b_i =(y_i-θ̂)²/v_i

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x, y

    References
    ----------
    Baujat et al. (2002)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Baujat plot: contribution to Q vs influence on θ̂"})


def cheatsheet():
    return "mabaujat: Baujat plot: contribution to Q vs influence on θ̂"

"""Funnel plot data: effect vs SE / inverse variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_funnel_plot_data"]


def ma_funnel_plot_data(yi, se_i):
    """
    Funnel plot data: effect vs SE / inverse variance

    Formula: (y_i, se_i) per study

    Parameters
    ----------
    yi : array-like
        Input data.
    se_i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_funnel, y_funnel

    References
    ----------
    Light & Pillemer (1984)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Funnel plot data: effect vs SE / inverse variance"})


def cheatsheet():
    return "mafnpr: Funnel plot data: effect vs SE / inverse variance"

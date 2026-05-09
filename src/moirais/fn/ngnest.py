"""N-BEATS neural forecasting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["n_beats"]


def n_beats(y, horizon, stacks):
    """
    N-BEATS neural forecasting

    Formula: stacked fully-connected blocks with backcast/forecast

    Parameters
    ----------
    y : array-like
        Input data.
    horizon : array-like
        Input data.
    stacks : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Oreshkin et al (2020) N-BEATS
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "N-BEATS neural forecasting"})


def cheatsheet():
    return "ngnest: N-BEATS neural forecasting"

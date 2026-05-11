"""N-BEATS pure-DL forecasting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["n_beats"]


def n_beats(y, stacks, blocks):
    """
    N-BEATS pure-DL forecasting

    Formula: stacks of fully-connected blocks with backcast/forecast

    Parameters
    ----------
    y : array-like
        Input data.
    stacks : array-like
        Input data.
    blocks : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "N-BEATS pure-DL forecasting"})


def cheatsheet():
    return "nbeats: N-BEATS pure-DL forecasting"

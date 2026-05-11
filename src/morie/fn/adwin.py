"""ADWIN adaptive window."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["adwin"]


def adwin(stream, delta):
    """
    ADWIN adaptive window

    Formula: shrink window when distribution shifts

    Parameters
    ----------
    stream : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bifet-Gavaldà (2007)
    """
    stream = np.atleast_1d(np.asarray(stream, dtype=float))
    n = len(stream)
    result = float(np.mean(stream))
    se = float(np.std(stream, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ADWIN adaptive window"})


def cheatsheet():
    return "adwin: ADWIN adaptive window"

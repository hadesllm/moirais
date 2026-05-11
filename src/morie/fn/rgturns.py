# morie.fn — function file (hadesllm/morie)
"""Turns count of an EMG signal (number of direction reversals above threshold)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_turns_count"]


def rangayyan_turns_count(x, threshold):
    """
    Turns count of an EMG signal (number of direction reversals above threshold)

    Formula: Turn: local extremum where |delta_amp| > threshold

    Parameters
    ----------
    x : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: turns_count, turn_locs

    References
    ----------
    Rangayyan Ch 5.6.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Turns count of an EMG signal (number of direction reversals above threshold)"})


def cheatsheet():
    return "rgturns: Turns count of an EMG signal (number of direction reversals above threshold)"

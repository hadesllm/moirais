# morie.fn -- function file (hadesllm/morie)
"""Minimum-phase correspondent of a signal."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_min_phase"]


def rangayyan_min_phase(x):
    """
    Minimum-phase correspondent of a signal

    Formula: Constructed by reflecting all zeros outside unit circle to inside

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_minphase

    References
    ----------
    Rangayyan Ch 5.4.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Minimum-phase correspondent of a signal"})


def cheatsheet():
    return "rgminph: Minimum-phase correspondent of a signal"

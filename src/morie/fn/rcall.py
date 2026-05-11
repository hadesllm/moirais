# morie.fn — function file (hadesllm/morie)
"""Roll call matrix analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["roll_call_analysis"]


def roll_call_analysis(x):
    """
    Roll call matrix analysis

    Formula: V = {v_ij}, v_ij in {yea, nay, absent}

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Armstrong Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Roll call matrix analysis"})


def cheatsheet():
    return "rcall: Roll call matrix analysis"

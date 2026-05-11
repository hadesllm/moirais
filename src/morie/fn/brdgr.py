# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bridge observations for cross-chamber comparison."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bridge_observations"]


def bridge_observations(x):
    """
    Bridge observations for cross-chamber comparison

    Formula: Link legislators across sessions via common votes

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
    Armstrong Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bridge observations for cross-chamber comparison"})


def cheatsheet():
    return "brdgr: Bridge observations for cross-chamber comparison"

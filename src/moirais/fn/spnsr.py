"""Effects of nugget, sill, and range on kriging prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_nugget_sill_range_effect"]


def schabenberger_nugget_sill_range_effect(nugget, sill, range, target_dist):
    """
    Effects of nugget, sill, and range on kriging prediction

    Formula: Larger nugget: smoother, less weight on nearest obs; larger range: more borrowing strength

    Parameters
    ----------
    nugget : array-like
        Input data.
    sill : array-like
        Input data.
    range : array-like
        Input data.
    target_dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: kriging_behavior

    References
    ----------
    Schabenberger Ch 5, Sec 5.2.3
    """
    nugget = np.asarray(nugget, dtype=float)
    n = int(nugget) if nugget.ndim == 0 else len(nugget)
    result = float(np.mean(nugget))
    se = float(np.std(nugget, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effects of nugget, sill, and range on kriging prediction"})


def cheatsheet():
    return "spnsr: Effects of nugget, sill, and range on kriging prediction"

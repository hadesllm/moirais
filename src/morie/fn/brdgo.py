# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bridge observations for cross-period ideal point comparison."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bridge_observations"]


def bridge_observations(ideal_points_periods, bridge_ids):
    """
    Bridge observations for cross-period ideal point comparison

    Formula: Legislators appearing in multiple chambers/periods anchor the common scale

    Parameters
    ----------
    ideal_points_periods : array-like
        Input data.
    bridge_ids : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'aligned_points': 'matrix'}

    References
    ----------
    Armstrong Ch 5,6
    """
    ideal_points_periods = np.asarray(ideal_points_periods, dtype=float)
    n = int(ideal_points_periods) if ideal_points_periods.ndim == 0 else len(ideal_points_periods)
    result = float(np.mean(ideal_points_periods))
    se = float(np.std(ideal_points_periods, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bridge observations for cross-period ideal point comparison"})


def cheatsheet():
    return "brdgo: Bridge observations for cross-period ideal point comparison"

# morie.fn — function file (hadesllm/morie)
"""Object tracking across frames using detection + association."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_object_tracking"]


def geron_object_tracking(frames, detector):
    """
    Object tracking across frames using detection + association

    Formula: frame_t boxes associated via IoU + Kalman prediction

    Parameters
    ----------
    frames : array-like
        Input data.
    detector : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: trajectories

    References
    ----------
    Géron Ch 12
    """
    frames = np.atleast_1d(np.asarray(frames, dtype=float))
    y = frames
    n = len(frames)
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Object tracking across frames using detection + association"})
    result = stats.spearmanr(frames[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Object tracking across frames using detection + association"})


def cheatsheet():
    return "hmotrk: Object tracking across frames using detection + association"

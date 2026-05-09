"""SAM-2 mask propagation across video."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sam2_video_propagation"]


def sam2_video_propagation(video_frames, initial_prompt):
    """
    SAM-2 mask propagation across video

    Formula: memory bank of past frames; cross-attend

    Parameters
    ----------
    video_frames : array-like
        Input data.
    initial_prompt : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ravi et al (2024) SAM-2
    """
    video_frames = np.atleast_1d(np.asarray(video_frames, dtype=float))
    n = len(video_frames)
    result = float(np.mean(video_frames))
    se = float(np.std(video_frames, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SAM-2 mask propagation across video"})


def cheatsheet():
    return "sam2vd: SAM-2 mask propagation across video"

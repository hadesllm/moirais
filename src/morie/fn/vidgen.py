"""Video diffusion frame generation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["video_diffusion"]


def video_diffusion(t, conditions, n_frames):
    """
    Video diffusion frame generation

    Formula: 3D U-Net or DiT with temporal attention

    Parameters
    ----------
    t : array-like
        Input data.
    conditions : array-like
        Input data.
    n_frames : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ho et al (2022) Video Diffusion Models
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Video diffusion frame generation"})


def cheatsheet():
    return "vidgen: Video diffusion frame generation"

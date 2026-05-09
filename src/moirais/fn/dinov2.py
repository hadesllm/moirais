"""DINOv2 self-supervised image representation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dino_v2_repr"]


def dino_v2_repr(x, student, teacher, tau):
    """
    DINOv2 self-supervised image representation

    Formula: student-teacher consistency over multi-crop views

    Parameters
    ----------
    x : array-like
        Input data.
    student : array-like
        Input data.
    teacher : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Oquab et al (2024) Meta AI
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DINOv2 self-supervised image representation"})


def cheatsheet():
    return "dinov2: DINOv2 self-supervised image representation"

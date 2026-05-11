"""OpenCLIP open-source training."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["open_clip"]


def open_clip(images, texts, batch):
    """
    OpenCLIP open-source training

    Formula: reproduce CLIP at scale on LAION

    Parameters
    ----------
    images : array-like
        Input data.
    texts : array-like
        Input data.
    batch : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cherti et al (2023) OpenCLIP
    """
    images = np.atleast_1d(np.asarray(images, dtype=float))
    n = len(images)
    result = float(np.mean(images))
    se = float(np.std(images, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "OpenCLIP open-source training"})


def cheatsheet():
    return "opnclp: OpenCLIP open-source training"

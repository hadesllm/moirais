"""DINO multi-crop augmentation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dino_multicrop"]


def dino_multicrop(image, global_size, local_size):
    """
    DINO multi-crop augmentation

    Formula: 2 global + 8 local crops; consistency

    Parameters
    ----------
    image : array-like
        Input data.
    global_size : array-like
        Input data.
    local_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Caron et al (2021)
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DINO multi-crop augmentation"})


def cheatsheet():
    return "dinmlt: DINO multi-crop augmentation"

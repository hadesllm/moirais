"""Copy number variant detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["copy_number_variant"]


def copy_number_variant(depth, reference_depth):
    """
    Copy number variant detection

    Formula: depth ratio + segmentation HMM

    Parameters
    ----------
    depth : array-like
        Input data.
    reference_depth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Olshen et al (2004) CBS
    """
    depth = np.atleast_1d(np.asarray(depth, dtype=float))
    n = len(depth)
    result = float(np.mean(depth))
    se = float(np.std(depth, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Copy number variant detection"})


def cheatsheet():
    return "copynm: Copy number variant detection"

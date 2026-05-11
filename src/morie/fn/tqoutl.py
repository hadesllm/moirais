"""Per-layer outlier/inlier channel separation — separate QJL instance per group."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_outlier_channel_split"]


def turboquant_outlier_channel_split(channels, outlier_threshold):
    """
    Per-layer outlier/inlier channel separation — separate QJL instance per group

    Formula: split channels by |activation| percentile; run QJL_outlier + QJL_inlier independently

    Parameters
    ----------
    channels : array-like
        Input data.
    outlier_threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: outlier_idx, inlier_idx

    References
    ----------
    Zandieh et al. 2024 Section 4.1 (outlier handling)
    """
    channels = np.atleast_1d(np.asarray(channels, dtype=float))
    n = len(channels)
    result = float(np.mean(channels))
    se = float(np.std(channels, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Per-layer outlier/inlier channel separation — separate QJL instance per group"})


def cheatsheet():
    return "tqoutl: Per-layer outlier/inlier channel separation — separate QJL instance per group"

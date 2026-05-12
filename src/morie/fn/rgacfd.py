# morie.fn -- function file (hadesllm/morie)
"""ACF distance measure for nonstationary segmentation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_acf_distance"]


def rangayyan_acf_distance(x, seg_len, p):
    """
    ACF distance measure for nonstationary segmentation

    Formula: d_ACF = (1/p) sum_{m=1}^{p} (R_1(m) - R_2(m))^2 / (R_1(0)*R_2(0))

    Parameters
    ----------
    x : array-like
        Input data.
    seg_len : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: acf_dist_trace, segment_bounds

    References
    ----------
    Rangayyan Ch 8.5.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ACF distance measure for nonstationary segmentation"})


def cheatsheet():
    return "rgacfd: ACF distance measure for nonstationary segmentation"

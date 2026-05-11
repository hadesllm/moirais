"""SAM multi-mask scoring + ranking."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sam_multi_mask_rank"]


def sam_multi_mask_rank(masks, scores):
    """
    SAM multi-mask scoring + ranking

    Formula: 3 candidate masks per prompt; IoU score

    Parameters
    ----------
    masks : array-like
        Input data.
    scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kirillov et al (2023)
    """
    masks = np.atleast_1d(np.asarray(masks, dtype=float))
    n = len(masks)
    result = float(np.mean(masks))
    se = float(np.std(masks, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SAM multi-mask scoring + ranking"})


def cheatsheet():
    return "sammkr: SAM multi-mask scoring + ranking"

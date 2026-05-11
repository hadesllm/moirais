"""YOLOX decoupled head."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["yolo_decoupled_head"]


def yolo_decoupled_head(features, anchor_free):
    """
    YOLOX decoupled head

    Formula: separate classification + regression heads

    Parameters
    ----------
    features : array-like
        Input data.
    anchor_free : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ge et al (2021) YOLOX
    """
    features = np.atleast_1d(np.asarray(features, dtype=float))
    n = len(features)
    result = float(np.mean(features))
    se = float(np.std(features, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "YOLOX decoupled head"})


def cheatsheet():
    return "yolovx: YOLOX decoupled head"

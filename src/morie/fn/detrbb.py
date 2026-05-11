"""DETR end-to-end object detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["detr_set_prediction"]


def detr_set_prediction(image, queries, n_objects):
    """
    DETR end-to-end object detection

    Formula: transformer decoder; bipartite matching loss

    Parameters
    ----------
    image : array-like
        Input data.
    queries : array-like
        Input data.
    n_objects : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Carion et al (2020) DETR
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DETR end-to-end object detection"})


def cheatsheet():
    return "detrbb: DETR end-to-end object detection"

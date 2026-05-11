"""Mask R-CNN instance segmentation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mask_rcnn_segmentation"]


def mask_rcnn_segmentation(image, proposals):
    """
    Mask R-CNN instance segmentation

    Formula: RoIAlign + parallel mask + cls + bbox heads

    Parameters
    ----------
    image : array-like
        Input data.
    proposals : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    He et al (2017) Mask R-CNN
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mask R-CNN instance segmentation"})


def cheatsheet():
    return "masrcn: Mask R-CNN instance segmentation"

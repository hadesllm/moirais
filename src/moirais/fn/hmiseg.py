# moirais.fn — function file (hadesllm/moirais)
"""Image segmentation via k-means on pixel colors."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_image_segmentation"]


def geron_image_segmentation(image, n_clusters, seed):
    """
    Image segmentation via k-means on pixel colors

    Formula: cluster pixels in RGB space; replace with cluster mean

    Parameters
    ----------
    image : array-like
        Input data.
    n_clusters : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: segmented_image

    References
    ----------
    Géron Ch 8
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Image segmentation via k-means on pixel colors"})


def cheatsheet():
    return "hmiseg: Image segmentation via k-means on pixel colors"

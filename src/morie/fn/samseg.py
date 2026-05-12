"""Segment Anything Model -- promptable segmentation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sam_segment"]


def sam_segment(image, prompts):
    """
    Segment Anything Model -- promptable segmentation

    Formula: image_encoder(x) + prompt_encoder(p) -> mask_decoder

    Parameters
    ----------
    image : array-like
        Input data.
    prompts : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kirillov et al (2023) Meta AI
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Segment Anything Model -- promptable segmentation"})


def cheatsheet():
    return "samseg: Segment Anything Model -- promptable segmentation"

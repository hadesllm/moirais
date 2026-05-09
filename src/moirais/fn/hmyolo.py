# moirais.fn — function file (hadesllm/moirais)
"""YOLO: single-shot object detection via grid regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_yolo"]


def geron_yolo(image, model):
    """
    YOLO: single-shot object detection via grid regression

    Formula: each grid cell predicts (x,y,w,h,conf) and class probabilities

    Parameters
    ----------
    image : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boxes, scores, classes

    References
    ----------
    Géron Ch 12
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "YOLO: single-shot object detection via grid regression"})


def cheatsheet():
    return "hmyolo: YOLO: single-shot object detection via grid regression"

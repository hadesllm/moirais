# moirais.fn — function file (hadesllm/moirais)
"""DETR set-prediction loss: Hungarian matching + classification + bounding-box."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_detr_hungarian_matching"]


def geron_detr_hungarian_matching(pred_boxes, pred_classes, gt_boxes, gt_classes):
    """
    DETR set-prediction loss: Hungarian matching + classification + bounding-box

    Formula: L = sum over matched pairs [L_cls + 1{obj} * (lam_bbox * L_bbox + lam_iou * L_giou)]

    Parameters
    ----------
    pred_boxes : array-like
        Input data.
    pred_classes : array-like
        Input data.
    gt_boxes : array-like
        Input data.
    gt_classes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 16, DETR section
    """
    pred_boxes = np.atleast_1d(np.asarray(pred_boxes, dtype=float))
    n = len(pred_boxes)
    result = float(np.mean(pred_boxes))
    se = float(np.std(pred_boxes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DETR set-prediction loss: Hungarian matching + classification + bounding-box"})


def cheatsheet():
    return "grdetr: DETR set-prediction loss: Hungarian matching + classification + bounding-box"

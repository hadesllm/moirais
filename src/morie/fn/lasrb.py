# morie.fn -- function file (hadesllm/morie)
"""IoU tracking metric. 'Eyes in the sky.' -- Laserbeak"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def iou_metric(
    boxes_pred: np.ndarray,
    boxes_true: np.ndarray,
) -> DescriptiveResult:
    """Compute Intersection over Union (IoU) between predicted and true boxes.

    Each box is [x1, y1, x2, y2] (top-left, bottom-right).

    Parameters
    ----------
    boxes_pred : ndarray of shape (n, 4)
        Predicted bounding boxes.
    boxes_true : ndarray of shape (n, 4)
        Ground truth bounding boxes.

    Returns
    -------
    DescriptiveResult
        With ``value`` = mean IoU and ``extra`` containing per-box IoU.
    """
    bp = np.asarray(boxes_pred, dtype=float)
    bt = np.asarray(boxes_true, dtype=float)
    if bp.ndim == 1:
        bp = bp.reshape(1, -1)
    if bt.ndim == 1:
        bt = bt.reshape(1, -1)
    if bp.shape[1] != 4 or bt.shape[1] != 4:
        raise ValueError("Boxes must have 4 columns [x1, y1, x2, y2]")
    if bp.shape[0] != bt.shape[0]:
        raise ValueError("Same number of predicted and true boxes required")

    x1 = np.maximum(bp[:, 0], bt[:, 0])
    y1 = np.maximum(bp[:, 1], bt[:, 1])
    x2 = np.minimum(bp[:, 2], bt[:, 2])
    y2 = np.minimum(bp[:, 3], bt[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_pred = (bp[:, 2] - bp[:, 0]) * (bp[:, 3] - bp[:, 1])
    area_true = (bt[:, 2] - bt[:, 0]) * (bt[:, 3] - bt[:, 1])
    union_area = area_pred + area_true - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    mean_iou = float(iou.mean())

    return DescriptiveResult(
        name="iou_metric",
        value=mean_iou,
        extra={"per_box_iou": iou, "n_boxes": len(iou), "min_iou": float(iou.min()), "max_iou": float(iou.max())},
    )


lasrb = iou_metric


def cheatsheet() -> str:
    return "iou_metric({}) -> IoU tracking metric. 'Eyes in the sky.' -- Laserbeak"

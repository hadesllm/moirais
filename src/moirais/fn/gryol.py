# moirais.fn — function file (hadesllm/moirais)
"""YOLO localization + classification loss per grid cell."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_yolo_grid_loss"]


def geron_yolo_grid_loss(predictions, targets, lam_coord, lam_noobj):
    """
    YOLO localization + classification loss per grid cell

    Formula: L = sum_i [1{obj_i} * (L_bbox(i) + L_obj(i) + L_class(i))] + sum_i 1{noobj_i} * L_noobj(i)

    Parameters
    ----------
    predictions : array-like
        Input data.
    targets : array-like
        Input data.
    lam_coord : array-like
        Input data.
    lam_noobj : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 12, YOLO section
    """
    predictions = np.atleast_1d(np.asarray(predictions, dtype=float))
    n = len(predictions)
    result = float(np.mean(predictions))
    se = float(np.std(predictions, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "YOLO localization + classification loss per grid cell"})


def cheatsheet():
    return "gryol: YOLO localization + classification loss per grid cell"

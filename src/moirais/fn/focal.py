# moirais.fn — function file (hadesllm/moirais)
r"""Focal loss for class imbalance.

Downweights easy examples, emphasizes hard negatives.

References
----------
Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
Focal loss for dense object detection.
In ICCV (pp. 2980-2988).
"""

__all__ = ["focal"]

import numpy as np


def focal(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    epsilon=1e-7,
):
    """
    Focal loss.

    Parameters
    ----------
    y_true : ndarray
        True binary labels, shape (n_samples,).
    y_pred : ndarray
        Predicted probabilities, shape (n_samples,).
    alpha : float, optional
        Weighting factor. Default 0.25.
    gamma : float, optional
        Exponent for modulating loss. Default 2.0.
    from_logits : bool, optional
        If True, apply sigmoid to y_pred. Default False.
    epsilon : float, optional
        Numerical stability. Default 1e-7.

    Returns
    -------
    float
        Mean focal loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    if from_logits:
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))

    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    ce_loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    modulation = (1 - np.abs(y_true - y_pred)) ** gamma

    focal_loss = alpha * modulation * ce_loss

    return float(np.mean(focal_loss))

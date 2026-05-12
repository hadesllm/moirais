# morie.fn -- function file (hadesllm/morie)
r"""Huber loss (robust regression).

Combines MSE (quadratic) and MAE (linear) losses.

References
----------
Huber, P. J. (1964).
Robust estimation of a location parameter.
Annals of Mathematical Statistics, 35(1), 73-101.
"""

__all__ = ["huber"]

import numpy as np


def huber(
    y_true,
    y_pred,
    delta=1.0,
):
    """
    Huber loss.

    Parameters
    ----------
    y_true : ndarray
        True values, shape (n_samples,).
    y_pred : ndarray
        Predicted values, shape (n_samples,).
    delta : float, optional
        Transition point. Default 1.0.

    Returns
    -------
    float
        Mean Huber loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    if delta <= 0:
        raise ValueError("delta must be positive")

    residual = y_true - y_pred
    abs_residual = np.abs(residual)

    loss = np.where(
        abs_residual <= delta,
        0.5 * residual**2,
        delta * (abs_residual - 0.5 * delta),
    )

    return float(np.mean(loss))

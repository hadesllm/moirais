# morie.fn — function file (hadesllm/morie)
"""Log loss (binary cross-entropy)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def log_loss(
    y_true: Union[np.ndarray, Any],
    y_prob: Union[np.ndarray, Any],
    *,
    eps: float = 1e-15,
) -> float:
    """Compute log loss (binary cross-entropy).

    L = -1/n * sum(y*log(p) + (1-y)*log(1-p))

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground-truth labels (0/1).
    y_prob : array-like of shape (n,)
        Predicted probabilities.
    eps : float
        Clipping epsilon to avoid log(0) (default 1e-15).

    Returns
    -------
    float
        Log loss value (lower is better).

    References
    ----------
    Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
        Springer. Section 4.3.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_prob, dtype=float).ravel()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_prob must have same length.")

    yp = np.clip(yp, eps, 1 - eps)
    return float(-np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp)))


logls = log_loss


def cheatsheet() -> str:
    return "log_loss({}) -> Log loss (binary cross-entropy)."

# morie.fn — function file (hadesllm/morie)
"""Nadaraya-Watson kernel regression. 'Knowing yourself is true wisdom. -- Lao Tzu'"""
from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kernel_regression(
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray | None = None,
    bandwidth: float = 1.0,
) -> DescriptiveResult:
    """Nadaraya-Watson kernel regression with Gaussian kernel.

    Parameters
    ----------
    x : ndarray
        Training x (1D).
    y : ndarray
        Training y.
    x_pred : ndarray or None
        Prediction points. Defaults to *x*.
    bandwidth : float, default 1.0

    Returns
    -------
    DescriptiveResult
        ``value`` is the predicted y array.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x_pred is None:
        x_pred = x.copy()
    x_pred = np.asarray(x_pred, dtype=float).ravel()

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    y_pred = np.empty(len(x_pred))
    for i, xp in enumerate(x_pred):
        w = np.exp(-0.5 * ((x - xp) / bandwidth) ** 2)
        w_sum = w.sum()
        y_pred[i] = (w @ y) / w_sum if w_sum > 0 else np.mean(y)

    return DescriptiveResult(
        name="Nadaraya-Watson",
        value=y_pred,
        extra={"bandwidth": bandwidth, "n_train": len(x), "n_pred": len(x_pred)},
    )


kreg = kernel_regression


def cheatsheet() -> str:
    return "kernel_regression({}) -> Nadaraya-Watson kernel regression. 'I have the high ground.'"

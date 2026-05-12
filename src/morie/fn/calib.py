# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Calibration curve and Brier score."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def calibration_curve(
    y_true: Union[np.ndarray, Any],
    y_prob: Union[np.ndarray, Any],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration curve (reliability diagram) and Brier score.

    Bins predicted probabilities and computes observed frequency per bin.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground-truth labels (0/1).
    y_prob : array-like of shape (n,)
        Predicted probabilities.
    n_bins : int
        Number of equal-width bins (default 10).

    Returns
    -------
    dict
        bin_means (predicted), bin_freqs (observed), brier_score.

    References
    ----------
    Brier, G. W. (1950). Verification of forecasts expressed in terms of
        probability. *Monthly Weather Review*, 78(1), 1-3.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_prob, dtype=float).ravel()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_prob must have same length.")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_means = []
    bin_freqs = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if lo == bin_edges[0]:
            mask = (yp >= lo) & (yp <= hi)
        else:
            mask = (yp > lo) & (yp <= hi)
        if mask.sum() > 0:
            bin_means.append(float(yp[mask].mean()))
            bin_freqs.append(float(yt[mask].mean()))

    brier = float(np.mean((yp - yt) ** 2))

    return {
        "bin_means": np.array(bin_means),
        "bin_freqs": np.array(bin_freqs),
        "brier_score": brier,
    }


calib = calibration_curve


def cheatsheet() -> str:
    return "calibration_curve({}) -> Calibration curve and Brier score."

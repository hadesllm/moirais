# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Calibration / reliability diagram."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There's always a bigger fish."


def calibration_plot(y_true, y_prob, n_bins=10, **kwargs) -> DescriptiveResult:
    """Compute calibration (reliability) diagram data.

    Bins predicted probabilities and computes mean predicted vs. actual
    fraction of positives per bin. Reports Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels (0/1).
    y_prob : array-like of shape (n,)
        Predicted probabilities in [0, 1].
    n_bins : int
        Number of bins (default 10).

    Returns
    -------
    DescriptiveResult
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means_pred = []
    bin_means_true = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        count = mask.sum()
        bin_counts.append(int(count))
        if count > 0:
            bin_means_pred.append(float(y_prob[mask].mean()))
            bin_means_true.append(float(y_true[mask].mean()))
        else:
            bin_means_pred.append(float("nan"))
            bin_means_true.append(float("nan"))

    bin_means_pred = np.array(bin_means_pred)
    bin_means_true = np.array(bin_means_true)
    bin_counts = np.array(bin_counts)

    valid = ~np.isnan(bin_means_pred)
    ece = float(np.sum(bin_counts[valid] * np.abs(bin_means_true[valid] - bin_means_pred[valid])) / len(y_true))

    return DescriptiveResult(
        name="calibration_plot",
        value=ece,
        extra={
            "bin_means_pred": bin_means_pred,
            "bin_means_true": bin_means_true,
            "bin_counts": bin_counts,
            "bin_edges": bin_edges,
            "ece": ece,
            "n_bins": n_bins,
        },
    )


calpl = calibration_plot


def cheatsheet() -> str:
    return "calibration_plot({}) -> Calibration / reliability diagram."

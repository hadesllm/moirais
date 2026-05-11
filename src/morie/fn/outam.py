# morie.fn — function file (hadesllm/morie)
"""Outlier detection in A-M residuals."""

from __future__ import annotations

from ._containers import DescriptiveResult


def outlier_detection_am(residuals, threshold: float = 2.0) -> DescriptiveResult:
    """Detect outlier respondents based on A-M residual magnitude.

    :param residuals: Residual matrix (n_resp x n_stim).
    :param threshold: Z-score threshold for outlier flag.
    :return: DescriptiveResult with outlier indices.

    .. epigraph:: "There are no miracles, only the inevitable." -- Yuuko, xxxHolic
    """
    import numpy as np

    R = np.asarray(residuals, dtype=float)
    row_rmse = np.sqrt(np.nanmean(R**2, axis=1))
    mu = np.nanmean(row_rmse)
    sd = np.nanstd(row_rmse)
    if sd == 0:
        outliers = []
    else:
        z = (row_rmse - mu) / sd
        outliers = np.where(np.abs(z) > threshold)[0].tolist()
    return DescriptiveResult(
        name="outlier_detection_am",
        value=len(outliers),
        extra={"outlier_indices": outliers, "row_rmse": row_rmse.tolist(), "threshold": threshold},
    )


outam = outlier_detection_am


def cheatsheet() -> str:
    return "outlier_detection_am({}) -> Outlier detection in A-M residuals."

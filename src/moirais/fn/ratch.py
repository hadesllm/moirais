# moirais.fn — function file (hadesllm/moirais)
"""Data cleaning / denoising pipeline. 'I can fix that!' -- Ratchet"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def repair_pipeline(
    data: pd.DataFrame | np.ndarray,
    *,
    clip_sigma: float = 3.0,
    fill_method: str = "median",
    smooth_window: int = 0,
) -> DescriptiveResult:
    """Apply a repair pipeline: outlier clipping, missing fill, optional smoothing.

    Parameters
    ----------
    data : DataFrame or ndarray
        Input data. If DataFrame, operates on all numeric columns.
    clip_sigma : float
        Clip values beyond this many standard deviations from the mean.
    fill_method : str
        How to fill NaN: 'median', 'mean', or 'zero'.
    smooth_window : int
        If > 0, apply rolling mean with this window size.

    Returns
    -------
    DescriptiveResult
        With ``value`` = cleaned data (DataFrame or ndarray) and
        ``extra`` containing counts of repaired values.
    """
    if fill_method not in ("median", "mean", "zero"):
        raise ValueError(f"fill_method must be median/mean/zero, got {fill_method}")

    is_df = isinstance(data, pd.DataFrame)
    if is_df:
        arr = data.select_dtypes(include="number").to_numpy(dtype=float).copy()
    else:
        arr = np.asarray(data, dtype=float).copy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

    n_clipped = 0
    n_filled = 0

    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue

        mu, sigma = valid.mean(), valid.std()
        if sigma > 0:
            mask = ~np.isnan(col) & (np.abs(col - mu) > clip_sigma * sigma)
            n_clipped += int(mask.sum())
            col[mask] = np.clip(col[mask], mu - clip_sigma * sigma, mu + clip_sigma * sigma)

        nan_mask = np.isnan(col)
        n_filled += int(nan_mask.sum())
        if fill_method == "median":
            fill_val = np.nanmedian(col)
        elif fill_method == "mean":
            fill_val = np.nanmean(col)
        else:
            fill_val = 0.0
        col[nan_mask] = fill_val

        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            col[:] = np.convolve(col, kernel, mode="same")

        arr[:, col_idx] = col

    if is_df:
        result_data = data.copy()
        num_cols = data.select_dtypes(include="number").columns
        result_data[num_cols] = arr
    else:
        result_data = arr.squeeze() if data.ndim == 1 else arr

    return DescriptiveResult(
        name="repair_pipeline",
        value=result_data,
        extra={"n_clipped": n_clipped, "n_filled": n_filled, "clip_sigma": clip_sigma, "fill_method": fill_method},
    )


ratch = repair_pipeline


def cheatsheet() -> str:
    return "repair_pipeline({}) -> Data cleaning / denoising pipeline. 'I can fix that!' -- Rat"

# moirais.fn — function file (hadesllm/moirais)
"""Wind rose / rose plot construction. 'I make my own choices.' -- Rose Wilson"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def wind_rose(
    data: pd.DataFrame,
    *,
    direction: str = "direction",
    speed: str = "speed",
    n_sectors: int = 16,
    speed_bins: list[float] | None = None,
) -> DescriptiveResult:
    """Construct wind rose data (frequency by direction and speed class).

    Parameters
    ----------
    data : DataFrame
        Must contain direction (degrees) and speed columns.
    direction : str
        Wind direction column (0-360 degrees, meteorological convention).
    speed : str
        Wind speed column.
    n_sectors : int
        Number of directional bins (e.g. 8, 16, 36).
    speed_bins : list of float or None
        Speed class boundaries.  Default [0, 2, 5, 10, 15, 20, inf].

    Returns
    -------
    DescriptiveResult
        ``value`` = dict with sector frequencies by speed class.
    """
    _validate_df(data, direction, speed)
    df = data[[direction, speed]].dropna()
    dirs = df[direction].to_numpy(dtype=float) % 360
    spds = df[speed].to_numpy(dtype=float)
    n = len(dirs)
    if n < 1:
        raise ValueError("No valid data")
    if speed_bins is None:
        speed_bins = [0, 2, 5, 10, 15, 20, float("inf")]
    sector_width = 360.0 / n_sectors
    sector_centers = np.arange(0, 360, sector_width)
    freq_table = np.zeros((n_sectors, len(speed_bins) - 1))
    for i, center in enumerate(sector_centers):
        lo = (center - sector_width / 2) % 360
        hi = (center + sector_width / 2) % 360
        if lo < hi:
            mask = (dirs >= lo) & (dirs < hi)
        else:
            mask = (dirs >= lo) | (dirs < hi)
        sector_spds = spds[mask]
        for j in range(len(speed_bins) - 1):
            freq_table[i, j] = int(np.sum((sector_spds >= speed_bins[j]) & (sector_spds < speed_bins[j + 1])))
    pct_table = freq_table / n * 100
    calm = float(np.sum(spds < speed_bins[1]) / n * 100) if len(speed_bins) > 1 else 0.0
    dominant_sector = int(np.argmax(freq_table.sum(axis=1)))
    return DescriptiveResult(
        name="Wind rose",
        value={
            "sector_centers": sector_centers.tolist(),
            "frequency": freq_table.tolist(),
            "percentage": pct_table.tolist(),
        },
        extra={
            "n": n,
            "n_sectors": n_sectors,
            "speed_bins": speed_bins,
            "calm_pct": round(calm, 1),
            "dominant_direction": float(sector_centers[dominant_sector]),
            "mean_speed": float(np.mean(spds)),
            "max_speed": float(np.max(spds)),
        },
    )


roslw = wind_rose


def cheatsheet() -> str:
    return "wind_rose({}) -> Wind rose / rose plot construction. 'I make my own choices.'"

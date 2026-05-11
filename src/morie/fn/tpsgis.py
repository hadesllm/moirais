"""Geographic crime density."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def tps_geo_analysis(
    lat: np.ndarray | list[float],
    lon: np.ndarray | list[float],
) -> DescriptiveResult:
    """Compute geographic crime density metrics.

    Calculates centroid, spread (std of coordinates), and bounding box.

    Parameters
    ----------
    lat : array-like
        Latitude values.
    lon : array-like
        Longitude values.

    Returns
    -------
    DescriptiveResult
    """
    la = np.asarray(lat, dtype=float)
    lo = np.asarray(lon, dtype=float)
    if len(la) != len(lo) or len(la) == 0:
        raise ValueError("lat and lon must be non-empty and same length")
    return DescriptiveResult(
        name="geo_crime_density",
        value=float(len(la)),
        extra={
            "centroid_lat": float(np.mean(la)),
            "centroid_lon": float(np.mean(lo)),
            "std_lat": float(np.std(la)),
            "std_lon": float(np.std(lo)),
            "bbox": {
                "min_lat": float(np.min(la)),
                "max_lat": float(np.max(la)),
                "min_lon": float(np.min(lo)),
                "max_lon": float(np.max(lo)),
            },
            "n_points": len(la),
        },
    )


tpsgis = tps_geo_analysis


def cheatsheet() -> str:
    return "tps_geo_analysis({}) -> Geographic crime density."

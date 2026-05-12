# morie.fn -- function file (hadesllm/morie)
"""All models are wrong, but some are useful. -- George E. P. Box"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def geo_summary(
    values: np.ndarray,
    regions: list[str],
) -> DescriptiveResult:
    """Per-region descriptive statistics for geographic mapping.

    Parameters
    ----------
    values : ndarray
        Numeric values corresponding to regions.
    regions : list[str]
        Region labels (same length as *values*).

    Returns
    -------
    DescriptiveResult
    """
    values = np.asarray(values, dtype=float)
    if len(values) != len(regions):
        raise ValueError("values and regions must have the same length")

    region_stats = {}
    unique_regions = sorted(set(regions))
    for r in unique_regions:
        mask = np.array([reg == r for reg in regions])
        v = values[mask]
        region_stats[r] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "n": int(len(v)),
        }

    return DescriptiveResult(
        name="Geographic summary",
        value=len(unique_regions),
        extra={
            "region_stats": region_stats,
            "overall_mean": float(np.mean(values)),
            "overall_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n_regions": len(unique_regions),
            "n_total": len(values),
        },
    )


holo_g = geo_summary


def cheatsheet() -> str:
    return "geo_summary({}) -> Geographic / choropleth summary helper. 'I've been waiting f"

# morie.fn — function file (hadesllm/morie)
"""
Lon/lat to projected coordinates

Category: GeoProcss
"""

import numpy as np


def lonlat(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Lon/lat to projected coordinates

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if coords is None:
        coords = np.random.default_rng(0).uniform(-180, 180, (n, 2))
    stat = float(np.mean(np.linalg.norm(coords, axis=1)))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n_points": len(coords), "source": source_crs, "target": target_crs},
    )


short = "lonlat"
alias = "lonlat"
quote = "Dedicate your hearts! -- Erwin"
lonlat = lonlat


def cheatsheet() -> str:
    return "lonlat({}) -> Lon/lat to projected coordinates"

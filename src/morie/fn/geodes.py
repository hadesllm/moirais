# morie.fn -- function file (hadesllm/morie)
"""
Geodesic distance calculation

Category: GeoProcss
"""

import numpy as np


def geodes(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Geodesic distance calculation

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


short = "geodes"
alias = "geodes"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
geodes = geodes


def cheatsheet() -> str:
    return "geodes({}) -> Geodesic distance calculation"

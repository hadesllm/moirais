# morie.fn -- function file (hadesllm/morie)
"""
Regular grid generation

Category: GeoProcss
"""

import numpy as np


def grdgen(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Regular grid generation

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


short = "grdgen"
alias = "grdgen"
quote = "Walk without rhythm. -- Fremen proverb"
grdgen = grdgen


def cheatsheet() -> str:
    return "grdgen({}) -> Regular grid generation"

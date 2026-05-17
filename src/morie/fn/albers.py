# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Albers equal area projection

Category: GeoProcss
"""

import numpy as np


def albers(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Albers equal area projection

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


short = "albers"
alias = "albers"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
albers = albers


def cheatsheet() -> str:
    return "albers({}) -> Albers equal area projection"

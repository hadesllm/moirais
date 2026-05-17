# morie.fn -- function file (hadesllm/morie)
"""
Lambert conformal conic

Category: GeoProcss
"""

import numpy as np


def lambrt(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Lambert conformal conic

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


short = "lambrt"
alias = "lambrt"
quote = "The measure of a man is what he does with power. -- Plato"
lambrt = lambrt


def cheatsheet() -> str:
    return "lambrt({}) -> Lambert conformal conic"

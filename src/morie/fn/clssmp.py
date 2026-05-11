# morie.fn — function file (hadesllm/morie)
"""
Cluster spatial sampling

Category: GeoProcss
"""

import numpy as np


def clssmp(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Cluster spatial sampling

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


short = "clssmp"
alias = "clssmp"
quote = "Science! -- Jesse Pinkman"
clssmp = clssmp


def cheatsheet() -> str:
    return "clssmp({}) -> Cluster spatial sampling"

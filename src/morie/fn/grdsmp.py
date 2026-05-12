# morie.fn -- function file (hadesllm/morie)
"""
Grid sampling spatial

Category: GeoProcss
"""

import numpy as np


def grdsmp(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Grid sampling spatial

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


short = "grdsmp"
alias = "grdsmp"
quote = "The world is cruel but beautiful. -- Mikasa"
grdsmp = grdsmp


def cheatsheet() -> str:
    return "grdsmp({}) -> Grid sampling spatial"

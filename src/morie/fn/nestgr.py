# morie.fn -- function file (hadesllm/morie)
"""
Nested multi-resolution grid

Category: GeoProcss
"""

import numpy as np


def nestgr(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Nested multi-resolution grid

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


short = "nestgr"
alias = "nestgr"
quote = "What is now proved was once only imagined. -- William Blake"
nestgr = nestgr


def cheatsheet() -> str:
    return "nestgr({}) -> Nested multi-resolution grid"

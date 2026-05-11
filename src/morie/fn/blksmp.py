# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""
Block spatial cross-validation

Category: GeoProcss
"""

import numpy as np


def blksmp(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Block spatial cross-validation

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


short = "blksmp"
alias = "blksmp"
quote = "A lesson without pain is meaningless. -- Edward"
blksmp = blksmp


def cheatsheet() -> str:
    return "blksmp({}) -> Block spatial cross-validation"

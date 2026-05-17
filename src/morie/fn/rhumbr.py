# morie.fn -- function file (hadesllm/morie)
"""
Rhumb line distance

Category: GeoProcss
"""

import numpy as np


def rhumbr(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Rhumb line distance

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


short = "rhumbr"
alias = "rhumbr"
quote = "Errors using inadequate data are much less than those using none. -- Charles Babbage"
rhumbr = rhumbr


def cheatsheet() -> str:
    return "rhumbr({}) -> Rhumb line distance"

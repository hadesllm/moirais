# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Balanced spatial sampling

Category: GeoProcss
"""

import numpy as np


def balsmp(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Balanced spatial sampling

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


short = "balsmp"
alias = "balsmp"
quote = "There is no royal road to geometry. -- Euclid"
balsmp = balsmp


def cheatsheet() -> str:
    return "balsmp({}) -> Balanced spatial sampling"

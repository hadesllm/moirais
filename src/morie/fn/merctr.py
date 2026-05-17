# morie.fn -- function file (hadesllm/morie)
"""
Mercator projection

Category: GeoProcss
"""

import numpy as np


def merctr(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Mercator projection

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


short = "merctr"
alias = "merctr"
quote = "It is not the strongest that survives, but the most adaptable. -- Charles Darwin"
merctr = merctr


def cheatsheet() -> str:
    return "merctr({}) -> Mercator projection"

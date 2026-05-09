# moirais.fn — function file (hadesllm/moirais)
"""
Hexagonal grid sampling

Category: GeoProcss
"""

import numpy as np


def hexsmp(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Hexagonal grid sampling

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


short = "hexsmp"
alias = "hexsmp"
quote = "I alone level up. -- Sung Jin-Woo"
hexsmp = hexsmp


def cheatsheet() -> str:
    return "hexsmp({}) -> Hexagonal grid sampling"

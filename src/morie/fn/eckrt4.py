# morie.fn -- function file (hadesllm/morie)
"""
Eckert IV projection

Category: GeoProcss
"""

import numpy as np


def eckrt4(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Eckert IV projection

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


short = "eckrt4"
alias = "eckrt4"
quote = "Get in the robot, Shinji! -- Misato"
eckrt4 = eckrt4


def cheatsheet() -> str:
    return "eckrt4({}) -> Eckert IV projection"

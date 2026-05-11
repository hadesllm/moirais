# morie.fn — function file (hadesllm/morie)
"""
Oversampling rare spatial events

Category: GeoProcss
"""

import numpy as np


def ovrsmp(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Oversampling rare spatial events

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


short = "ovrsmp"
alias = "ovrsmp"
quote = "I mustn't run away. -- Shinji"
ovrsmp = ovrsmp


def cheatsheet() -> str:
    return "ovrsmp({}) -> Oversampling rare spatial events"

# morie.fn -- function file (hadesllm/morie)
"""
Mollweide projection

Category: GeoProcss
"""

import numpy as np


def mollwd(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Mollweide projection

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


short = "mollwd"
alias = "mollwd"
quote = "Logic is the foundation of all certain knowledge. -- Leonhard Euler"
mollwd = mollwd


def cheatsheet() -> str:
    return "mollwd({}) -> Mollweide projection"

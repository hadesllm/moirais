"""
Sinusoidal projection

Category: GeoProcss
"""

import numpy as np


def sinusd(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Sinusoidal projection

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


short = "sinusd"
alias = "sinusd"
quote = "There is no royal road to geometry. -- Euclid"
sinusd = sinusd


def cheatsheet() -> str:
    return "sinusd({}) -> Sinusoidal projection"

"""
Spatiotemporal sampling

Category: GeoProcss
"""

import numpy as np


def sptsmp(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Spatiotemporal sampling

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


short = "sptsmp"
alias = "sptsmp"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
sptsmp = sptsmp


def cheatsheet() -> str:
    return "sptsmp({}) -> Spatiotemporal sampling"

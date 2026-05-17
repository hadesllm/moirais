"""
Undersampling common spatial areas

Category: GeoProcss
"""

import numpy as np


def undsmp(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Undersampling common spatial areas

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


short = "undsmp"
alias = "undsmp"
quote = "Measure what is measurable, and make measurable what is not. -- Galileo Galilei"
undsmp = undsmp


def cheatsheet() -> str:
    return "undsmp({}) -> Undersampling common spatial areas"

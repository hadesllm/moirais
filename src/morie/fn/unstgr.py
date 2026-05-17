"""
Unstructured mesh generation

Category: GeoProcss
"""

import numpy as np


def unstgr(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Unstructured mesh generation

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


short = "unstgr"
alias = "unstgr"
quote = "I think, therefore I am. -- Rene Descartes"
unstgr = unstgr


def cheatsheet() -> str:
    return "unstgr({}) -> Unstructured mesh generation"

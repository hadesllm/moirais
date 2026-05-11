"""
Temporal stratified sampling

Category: GeoProcss
"""

import numpy as np


def tmpsmp(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Temporal stratified sampling

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


short = "tmpsmp"
alias = "tmpsmp"
quote = "You should enjoy the detours. -- Ging"
tmpsmp = tmpsmp


def cheatsheet() -> str:
    return "tmpsmp({}) -> Temporal stratified sampling"

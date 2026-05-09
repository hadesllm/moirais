# moirais.fn — function file (hadesllm/moirais)
"""
Random spatial sampling

Category: GeoProcss
"""

import numpy as np


def rndsmp(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Random spatial sampling

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


short = "rndsmp"
alias = "rndsmp"
quote = "Chaos is a ladder. -- Littlefinger"
rndsmp = rndsmp


def cheatsheet() -> str:
    return "rndsmp({}) -> Random spatial sampling"

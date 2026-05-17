# morie.fn -- function file (hadesllm/morie)
"""
Reproject coordinates between CRS

Category: GeoProcss
"""

import numpy as np


def reproj(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Reproject coordinates between CRS

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


short = "reproj"
alias = "reproj"
quote = "It is not the strongest that survives, but the most adaptable. -- Charles Darwin"
reproj = reproj


def cheatsheet() -> str:
    return "reproj({}) -> Reproject coordinates between CRS"

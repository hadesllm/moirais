# morie.fn -- function file (hadesllm/morie)
"""
Sprawl index spatial

Category: AreaGeo
"""

import numpy as np


def agspr(areas=None, perimeters=None, values=None, n=30):
    """Sprawl index spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if areas is None:
        areas = np.random.default_rng(0).uniform(10, 1000, n)
    if perimeters is None:
        perimeters = 4 * np.sqrt(areas) * np.random.default_rng(1).uniform(0.9, 1.5, n)
    if values is None:
        values = np.random.default_rng(2).standard_normal(n)
    stat = float(np.mean(4 * np.pi * areas / perimeters**2))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "n_patches": len(areas),
            "mean_area": float(np.mean(areas)),
            "mean_perimeter": float(np.mean(perimeters)),
            "compactness": float(stat),
        },
    )


short = "agspr"
alias = "agspr"
quote = "Desert power. -- Paul Muad'Dib"
agspr = agspr


def cheatsheet() -> str:
    return "agspr({}) -> Sprawl index spatial"

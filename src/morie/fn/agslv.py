# morie.fn -- function file (hadesllm/morie)
"""
Silviculture landscape metric

Category: AreaGeo
"""

import numpy as np


def agslv(areas=None, perimeters=None, values=None, n=30):
    """Silviculture landscape metric

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


short = "agslv"
alias = "agslv"
quote = "To understand God's thoughts we must study statistics. -- Florence Nightingale"
agslv = agslv


def cheatsheet() -> str:
    return "agslv({}) -> Silviculture landscape metric"

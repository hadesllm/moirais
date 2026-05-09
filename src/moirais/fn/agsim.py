# moirais.fn — function file (hadesllm/moirais)
"""
Similarity index patch

Category: AreaGeo
"""

import numpy as np


def agsim(areas=None, perimeters=None, values=None, n=30):
    """Similarity index patch

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


short = "agsim"
alias = "agsim"
quote = "Arise. -- Shadow Monarch"
agsim = agsim


def cheatsheet() -> str:
    return "agsim({}) -> Similarity index patch"

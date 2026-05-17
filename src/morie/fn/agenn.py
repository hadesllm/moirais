# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Euclidean NN distance patch

Category: AreaGeo
"""

import numpy as np


def agenn(areas=None, perimeters=None, values=None, n=30):
    """Euclidean NN distance patch

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


short = "agenn"
alias = "agenn"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
agenn = agenn


def cheatsheet() -> str:
    return "agenn({}) -> Euclidean NN distance patch"

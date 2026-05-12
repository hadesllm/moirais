# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Edge density

Category: AreaGeo
"""

import numpy as np


def aged(areas=None, perimeters=None, values=None, n=30):
    """Edge density

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


short = "aged"
alias = "aged"
quote = "One is all, all is one. -- Izumi"
aged = aged


def cheatsheet() -> str:
    return "aged({}) -> Edge density"

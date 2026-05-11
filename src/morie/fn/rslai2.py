# morie.fn — function file (hadesllm/morie)
"""
LAI from remote sensing

Category: RemSens
"""

import numpy as np


def rslai2(pixels=None, bands=None, n=100):
    """LAI from remote sensing

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if pixels is None:
        pixels = np.random.default_rng(0).uniform(0, 10000, (n, 4))
    if bands is None:
        bands = ["blue", "green", "red", "nir"]
    ndvi = (pixels[:, 3] - pixels[:, 2]) / (pixels[:, 3] + pixels[:, 2] + 1e-10)
    stat = float(np.mean(ndvi))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n_pixels": len(pixels), "n_bands": len(bands), "mean_ndvi": float(np.mean(ndvi))},
    )


short = "rslai2"
alias = "rslai2"
quote = "The world is cruel but beautiful. -- Mikasa"
rslai2 = rslai2


def cheatsheet() -> str:
    return "rslai2({}) -> LAI from remote sensing"

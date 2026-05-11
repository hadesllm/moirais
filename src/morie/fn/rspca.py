# morie.fn — function file (hadesllm/morie)
"""
PCA remote sensing

Category: RemSens
"""

import numpy as np


def rspca(pixels=None, bands=None, n=100):
    """PCA remote sensing

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


short = "rspca"
alias = "rspca"
quote = "Hold the door. -- Hodor"
rspca = rspca


def cheatsheet() -> str:
    return "rspca({}) -> PCA remote sensing"

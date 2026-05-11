# morie.fn — function file (hadesllm/morie)
"""
Radiometric terrain correction

Category: RemSens
"""

import numpy as np


def rsrtx(pixels=None, bands=None, n=100):
    """Radiometric terrain correction

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


short = "rsrtx"
alias = "rsrtx"
quote = "Power is everything. -- Sung Jin-Woo"
rsrtx = rsrtx


def cheatsheet() -> str:
    return "rsrtx({}) -> Radiometric terrain correction"

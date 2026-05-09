# moirais.fn — function file (hadesllm/moirais)
"""
Red-edge NDVI

Category: RemSens
"""

import numpy as np


def rsreg(pixels=None, bands=None, n=100):
    """Red-edge NDVI

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


short = "rsreg"
alias = "rsreg"
quote = "I am the one who knocks. -- Walter White"
rsreg = rsreg


def cheatsheet() -> str:
    return "rsreg({}) -> Red-edge NDVI"

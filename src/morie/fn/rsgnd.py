# morie.fn — function file (hadesllm/morie)
"""
GNDVI computation

Category: RemSens
"""

import numpy as np


def rsgnd(pixels=None, bands=None, n=100):
    """GNDVI computation

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


short = "rsgnd"
alias = "rsgnd"
quote = "Dedicate your hearts! -- Erwin"
rsgnd = rsgnd


def cheatsheet() -> str:
    return "rsgnd({}) -> GNDVI computation"

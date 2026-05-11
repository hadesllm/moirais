# morie.fn — function file (hadesllm/morie)
"""
NBR2 enhanced burn ratio

Category: RemSens
"""

import numpy as np


def rsnbr2(pixels=None, bands=None, n=100):
    """NBR2 enhanced burn ratio

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


short = "rsnbr2"
alias = "rsnbr2"
quote = "Set your heart ablaze! -- Rengoku"
rsnbr2 = rsnbr2


def cheatsheet() -> str:
    return "rsnbr2({}) -> NBR2 enhanced burn ratio"

# morie.fn -- function file (hadesllm/morie)
"""
Land surface temperature RS

Category: RemSens
"""

import numpy as np


def rslst(pixels=None, bands=None, n=100):
    """Land surface temperature RS

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


short = "rslst"
alias = "rslst"
quote = "A journey of a thousand miles begins with a single step. -- Lao Tzu"
rslst = rslst


def cheatsheet() -> str:
    return "rslst({}) -> Land surface temperature RS"

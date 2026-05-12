# morie.fn -- function file (hadesllm/morie)
"""
Confusion matrix RS

Category: RemSens
"""

import numpy as np


def rscnf(pixels=None, bands=None, n=100):
    """Confusion matrix RS

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


short = "rscnf"
alias = "rscnf"
quote = "Valar Morghulis. -- Braavos"
rscnf = rscnf


def cheatsheet() -> str:
    return "rscnf({}) -> Confusion matrix RS"

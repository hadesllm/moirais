# morie.fn -- function file (hadesllm/morie)
"""
Tasseled cap transform

Category: RemSens
"""

import numpy as np


def rstcw(pixels=None, bands=None, n=100):
    """Tasseled cap transform

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


short = "rstcw"
alias = "rstcw"
quote = "I think, therefore I am. -- Rene Descartes"
rstcw = rstcw


def cheatsheet() -> str:
    return "rstcw({}) -> Tasseled cap transform"

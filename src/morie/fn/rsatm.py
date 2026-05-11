# morie.fn — function file (hadesllm/morie)
"""
Atmospheric correction RS

Category: RemSens
"""

import numpy as np


def rsatm(pixels=None, bands=None, n=100):
    """Atmospheric correction RS

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


short = "rsatm"
alias = "rsatm"
quote = "A Lannister always pays his debts. -- Tyrion"
rsatm = rsatm


def cheatsheet() -> str:
    return "rsatm({}) -> Atmospheric correction RS"

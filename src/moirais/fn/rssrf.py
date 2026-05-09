# moirais.fn — function file (hadesllm/moirais)
"""
Surface reflectance

Category: RemSens
"""

import numpy as np


def rssrf(pixels=None, bands=None, n=100):
    """Surface reflectance

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


short = "rssrf"
alias = "rssrf"
quote = "Growing old is a blessing. -- Rengoku"
rssrf = rssrf


def cheatsheet() -> str:
    return "rssrf({}) -> Surface reflectance"

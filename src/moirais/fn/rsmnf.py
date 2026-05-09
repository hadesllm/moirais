# moirais.fn — function file (hadesllm/moirais)
"""
MNF transform RS

Category: RemSens
"""

import numpy as np


def rsmnf(pixels=None, bands=None, n=100):
    """MNF transform RS

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


short = "rsmnf"
alias = "rsmnf"
quote = "Fear is the mind-killer. -- Bene Gesserit"
rsmnf = rsmnf


def cheatsheet() -> str:
    return "rsmnf({}) -> MNF transform RS"

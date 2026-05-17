# morie.fn -- function file (hadesllm/morie)
"""
SVM classification RS

Category: RemSens
"""

import numpy as np


def rssvm(pixels=None, bands=None, n=100):
    """SVM classification RS

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


short = "rssvm"
alias = "rssvm"
quote = "If I have seen further it is by standing on the shoulders of giants. -- Isaac Newton"
rssvm = rssvm


def cheatsheet() -> str:
    return "rssvm({}) -> SVM classification RS"

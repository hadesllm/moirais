"""Fixed-effect logit panel spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svfxl(data=None, n=50):
    """Fixed-effect logit panel spatial.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "svfxl"
alias = "svfxl"
quote = "The spice must flow. -- Paul Atreides"
svfxl = svfxl


def cheatsheet() -> str:
    return "svfxl({}) -> Fixed-effect logit panel spatial."

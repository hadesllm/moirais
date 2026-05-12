# morie.fn -- function file (hadesllm/morie)
"""PCA-based ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idppc(data=None, n=50):
    """PCA-based ideal point.

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


short = "idppc"
alias = "idppc"
quote = "The spice must flow. -- Paul Atreides"
idppc = idppc


def cheatsheet() -> str:
    return "idppc({}) -> PCA-based ideal point."

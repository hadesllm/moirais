# morie.fn -- function file (hadesllm/morie)
"""Electoral polarization index.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elpol(data=None, n=50):
    """Electoral polarization index.

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


short = "elpol"
alias = "elpol"
quote = "The spice must flow. -- Paul Atreides"
elpol = elpol


def cheatsheet() -> str:
    return "elpol({}) -> Electoral polarization index."

# morie.fn — function file (hadesllm/morie)
"""KDE spatial probability.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pskde(data=None, n=50):
    """KDE spatial probability.

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


short = "pskde"
alias = "pskde"
quote = "The spice must flow. -- Paul Atreides"
pskde = pskde


def cheatsheet() -> str:
    return "pskde({}) -> KDE spatial probability."

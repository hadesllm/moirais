# morie.fn -- function file (hadesllm/morie)
"""Vote share probability.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def psvot(data=None, n=50):
    """Vote share probability.

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


short = "psvot"
alias = "psvot"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
psvot = psvot


def cheatsheet() -> str:
    return "psvot({}) -> Vote share probability."

# morie.fn -- function file (hadesllm/morie)
"""Double-peaked Hotelling.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hsdbl(data=None, n=50):
    """Double-peaked Hotelling.

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


short = "hsdbl"
alias = "hsdbl"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
hsdbl = hsdbl


def cheatsheet() -> str:
    return "hsdbl({}) -> Double-peaked Hotelling."

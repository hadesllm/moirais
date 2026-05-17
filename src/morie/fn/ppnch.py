# morie.fn -- function file (hadesllm/morie)
"""Party niche positioning.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppnch(data=None, n=50):
    """Party niche positioning.

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


short = "ppnch"
alias = "ppnch"
quote = "Number rules the universe. -- Pythagoras"
ppnch = ppnch


def cheatsheet() -> str:
    return "ppnch({}) -> Party niche positioning."

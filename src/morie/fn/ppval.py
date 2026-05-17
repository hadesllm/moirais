# morie.fn -- function file (hadesllm/morie)
"""Party valence positioning.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppval(data=None, n=50):
    """Party valence positioning.

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


short = "ppval"
alias = "ppval"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
ppval = ppval


def cheatsheet() -> str:
    return "ppval({}) -> Party valence positioning."

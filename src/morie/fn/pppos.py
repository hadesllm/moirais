# morie.fn -- function file (hadesllm/morie)
"""Party position estimation.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pppos(data=None, n=50):
    """Party position estimation.

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


short = "pppos"
alias = "pppos"
quote = "Logic is the foundation of all certain knowledge. -- Leonhard Euler"
pppos = pppos


def cheatsheet() -> str:
    return "pppos({}) -> Party position estimation."

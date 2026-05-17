# morie.fn -- function file (hadesllm/morie)
"""Party salience weighting.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppsal(data=None, n=50):
    """Party salience weighting.

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


short = "ppsal"
alias = "ppsal"
quote = "Number rules the universe. -- Pythagoras"
ppsal = ppsal


def cheatsheet() -> str:
    return "ppsal({}) -> Party salience weighting."

# morie.fn -- function file (hadesllm/morie)
"""Roll call vote probability basic.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rcvot(data=None, n=50):
    """Roll call vote probability basic.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(data)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "rcvot"
alias = "rcvot"
quote = "Number rules the universe. -- Pythagoras"
rcvot = rcvot


def cheatsheet() -> str:
    return "rcvot({}) -> Roll call vote probability basic."

# morie.fn -- function file (hadesllm/morie)
"""Amendment committee procedure.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def cmamn(data=None, n=50):
    """Amendment committee procedure.

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


short = "cmamn"
alias = "cmamn"
quote = "Number rules the universe. -- Pythagoras"
cmamn = cmamn


def cheatsheet() -> str:
    return "cmamn({}) -> Amendment committee procedure."

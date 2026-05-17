# morie.fn -- function file (hadesllm/morie)
"""Qualified majority committee.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def cmqrm(data=None, n=50):
    """Qualified majority committee.

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


short = "cmqrm"
alias = "cmqrm"
quote = "Number rules the universe. -- Pythagoras"
cmqrm = cmqrm


def cheatsheet() -> str:
    return "cmqrm({}) -> Qualified majority committee."

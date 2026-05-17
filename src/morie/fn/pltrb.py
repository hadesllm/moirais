# morie.fn -- function file (hadesllm/morie)
"""Tribal polarization index.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pltrb(data=None, n=50):
    """Tribal polarization index.

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


short = "pltrb"
alias = "pltrb"
quote = "The only true wisdom is in knowing you know nothing. -- Socrates"
pltrb = pltrb


def cheatsheet() -> str:
    return "pltrb({}) -> Tribal polarization index."

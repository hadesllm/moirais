# morie.fn -- function file (hadesllm/morie)
"""Manifesto-based positioning.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppman(data=None, n=50):
    """Manifesto-based positioning.

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


short = "ppman"
alias = "ppman"
quote = "The spice must flow. -- Paul Atreides"
ppman = ppman


def cheatsheet() -> str:
    return "ppman({}) -> Manifesto-based positioning."

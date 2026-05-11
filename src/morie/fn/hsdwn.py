# morie.fn — function file (hadesllm/morie)
"""Downs electoral competition.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hsdwn(data=None, n=50):
    """Downs electoral competition.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(np.asarray(data, dtype=float))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "hsdwn"
alias = "hsdwn"
quote = "The spice must flow. -- Paul Atreides"
hsdwn = hsdwn


def cheatsheet() -> str:
    return "hsdwn({}) -> Downs electoral competition."

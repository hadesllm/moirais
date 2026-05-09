# moirais.fn — function file (hadesllm/moirais)
"""Multidimensional median voter set.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtms(data=None, n=50):
    """Multidimensional median voter set.

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


short = "mvtms"
alias = "mvtms"
quote = "The spice must flow. -- Paul Atreides"
mvtms = mvtms


def cheatsheet() -> str:
    return "mvtms({}) -> Multidimensional median voter set."

# moirais.fn — function file (hadesllm/moirais)
"""Plurality median voter.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtpl(data=None, n=50):
    """Plurality median voter.

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


short = "mvtpl"
alias = "mvtpl"
quote = "The spice must flow. -- Paul Atreides"
mvtpl = mvtpl


def cheatsheet() -> str:
    return "mvtpl({}) -> Plurality median voter."

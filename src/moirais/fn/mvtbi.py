# moirais.fn — function file (hadesllm/moirais)
"""Bicameral median voter.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtbi(data=None, n=50):
    """Bicameral median voter.

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


short = "mvtbi"
alias = "mvtbi"
quote = "The spice must flow. -- Paul Atreides"
mvtbi = mvtbi


def cheatsheet() -> str:
    return "mvtbi({}) -> Bicameral median voter."

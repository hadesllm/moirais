# moirais.fn — function file (hadesllm/moirais)
"""Party change positioning.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppchg(data=None, n=50):
    """Party change positioning.

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


short = "ppchg"
alias = "ppchg"
quote = "The spice must flow. -- Paul Atreides"
ppchg = ppchg


def cheatsheet() -> str:
    return "ppchg({}) -> Party change positioning."

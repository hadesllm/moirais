# moirais.fn — function file (hadesllm/moirais)
"""Party divergence dynamics.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppdvg(data=None, n=50):
    """Party divergence dynamics.

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


short = "ppdvg"
alias = "ppdvg"
quote = "The spice must flow. -- Paul Atreides"
ppdvg = ppdvg


def cheatsheet() -> str:
    return "ppdvg({}) -> Party divergence dynamics."

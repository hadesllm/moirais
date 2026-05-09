# moirais.fn — function file (hadesllm/moirais)
"""Dynamic polarization trend.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pldyn(data=None, n=50):
    """Dynamic polarization trend.

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


short = "pldyn"
alias = "pldyn"
quote = "The spice must flow. -- Paul Atreides"
pldyn = pldyn


def cheatsheet() -> str:
    return "pldyn({}) -> Dynamic polarization trend."

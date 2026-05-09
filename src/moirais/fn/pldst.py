# moirais.fn — function file (hadesllm/moirais)
"""Distance-based polarization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pldst(data=None, n=50):
    """Distance-based polarization.

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


short = "pldst"
alias = "pldst"
quote = "The spice must flow. -- Paul Atreides"
pldst = pldst


def cheatsheet() -> str:
    return "pldst({}) -> Distance-based polarization."

# moirais.fn — function file (hadesllm/moirais)
"""Mass-elite polarization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plmss(data=None, n=50):
    """Mass-elite polarization.

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


short = "plmss"
alias = "plmss"
quote = "The spice must flow. -- Paul Atreides"
plmss = plmss


def cheatsheet() -> str:
    return "plmss({}) -> Mass-elite polarization."

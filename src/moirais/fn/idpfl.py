# moirais.fn — function file (hadesllm/moirais)
"""Floor median ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpfl(data=None, n=50):
    """Floor median ideal point.

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


short = "idpfl"
alias = "idpfl"
quote = "The spice must flow. -- Paul Atreides"
idpfl = idpfl


def cheatsheet() -> str:
    return "idpfl({}) -> Floor median ideal point."

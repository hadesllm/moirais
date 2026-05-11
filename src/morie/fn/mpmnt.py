# morie.fn — function file (hadesllm/morie)
"""Party movement spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mpmnt(data=None, n=50):
    """Party movement spatial.

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


short = "mpmnt"
alias = "mpmnt"
quote = "The spice must flow. -- Paul Atreides"
mpmnt = mpmnt


def cheatsheet() -> str:
    return "mpmnt({}) -> Party movement spatial."

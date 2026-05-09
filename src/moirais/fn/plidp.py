# moirais.fn — function file (hadesllm/moirais)
"""Ideological polarization index.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plidp(data=None, n=50):
    """Ideological polarization index.

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


short = "plidp"
alias = "plidp"
quote = "The spice must flow. -- Paul Atreides"
plidp = plidp


def cheatsheet() -> str:
    return "plidp({}) -> Ideological polarization index."

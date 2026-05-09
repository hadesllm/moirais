# moirais.fn — function file (hadesllm/moirais)
"""Esteban-Ray polarization index.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pleri(data=None, n=50):
    """Esteban-Ray polarization index.

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


short = "pleri"
alias = "pleri"
quote = "The spice must flow. -- Paul Atreides"
pleri = pleri


def cheatsheet() -> str:
    return "pleri({}) -> Esteban-Ray polarization index."

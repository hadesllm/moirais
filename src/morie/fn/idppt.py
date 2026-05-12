# morie.fn -- function file (hadesllm/morie)
"""Pivot ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idppt(data=None, n=50):
    """Pivot ideal point.

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


short = "idppt"
alias = "idppt"
quote = "The spice must flow. -- Paul Atreides"
idppt = idppt


def cheatsheet() -> str:
    return "idppt({}) -> Pivot ideal point."

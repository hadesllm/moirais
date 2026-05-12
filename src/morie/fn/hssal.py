# morie.fn -- function file (hadesllm/morie)
"""Salience Hotelling model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hssal(data=None, n=50):
    """Salience Hotelling model.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(np.asarray(data, dtype=float))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "hssal"
alias = "hssal"
quote = "The spice must flow. -- Paul Atreides"
hssal = hssal


def cheatsheet() -> str:
    return "hssal({}) -> Salience Hotelling model."

# morie.fn -- function file (hadesllm/morie)
"""Salience electoral model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elsal(data=None, n=50):
    """Salience electoral model.

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


short = "elsal"
alias = "elsal"
quote = "The spice must flow. -- Paul Atreides"
elsal = elsal


def cheatsheet() -> str:
    return "elsal({}) -> Salience electoral model."

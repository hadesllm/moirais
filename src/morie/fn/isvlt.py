# morie.fn — function file (hadesllm/morie)
"""Valence-salience joint model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def isvlt(data=None, n=50):
    """Valence-salience joint model.

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


short = "isvlt"
alias = "isvlt"
quote = "The spice must flow. -- Paul Atreides"
isvlt = isvlt


def cheatsheet() -> str:
    return "isvlt({}) -> Valence-salience joint model."

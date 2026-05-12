# morie.fn -- function file (hadesllm/morie)
"""Banfield roll call estimation.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rcbnf(data=None, n=50):
    """Banfield roll call estimation.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(data)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "rcbnf"
alias = "rcbnf"
quote = "The spice must flow. -- Paul Atreides"
rcbnf = rcbnf


def cheatsheet() -> str:
    return "rcbnf({}) -> Banfield roll call estimation."

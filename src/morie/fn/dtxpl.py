# morie.fn -- function file (hadesllm/morie)
"""Exploratory dimensionality test.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtxpl(data=None, n=50):
    """Exploratory dimensionality test.

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


short = "dtxpl"
alias = "dtxpl"
quote = "The spice must flow. -- Paul Atreides"
dtxpl = dtxpl


def cheatsheet() -> str:
    return "dtxpl({}) -> Exploratory dimensionality test."

# morie.fn -- function file (hadesllm/morie)
"""Bayesian spatial probability.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def psbay(data=None, n=50):
    """Bayesian spatial probability.

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


short = "psbay"
alias = "psbay"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
psbay = psbay


def cheatsheet() -> str:
    return "psbay({}) -> Bayesian spatial probability."

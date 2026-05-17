# morie.fn -- function file (hadesllm/morie)
"""Uniform perturbation spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def psunf(data=None, n=50):
    """Uniform perturbation spatial.

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


short = "psunf"
alias = "psunf"
quote = "It is not what happens to you, but how you react, that matters. -- Epictetus"
psunf = psunf


def cheatsheet() -> str:
    return "psunf({}) -> Uniform perturbation spatial."

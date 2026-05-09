# moirais.fn — function file (hadesllm/moirais)
"""Gamma perturbation spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def psgmb(data=None, n=50):
    """Gamma perturbation spatial.

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


short = "psgmb"
alias = "psgmb"
quote = "The spice must flow. -- Paul Atreides"
psgmb = psgmb


def cheatsheet() -> str:
    return "psgmb({}) -> Gamma perturbation spatial."

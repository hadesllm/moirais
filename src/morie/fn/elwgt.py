# morie.fn — function file (hadesllm/morie)
"""Weighted electoral model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elwgt(data=None, n=50):
    """Weighted electoral model.

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


short = "elwgt"
alias = "elwgt"
quote = "The spice must flow. -- Paul Atreides"
elwgt = elwgt


def cheatsheet() -> str:
    return "elwgt({}) -> Weighted electoral model."

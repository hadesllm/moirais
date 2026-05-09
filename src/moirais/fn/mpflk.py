# moirais.fn — function file (hadesllm/moirais)
"""Flanking party model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mpflk(data=None, n=50):
    """Flanking party model.

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


short = "mpflk"
alias = "mpflk"
quote = "The spice must flow. -- Paul Atreides"
mpflk = mpflk


def cheatsheet() -> str:
    return "mpflk({}) -> Flanking party model."

"""Directional voting model (Rabinowitz-Macdonald).

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svdir(data=None, n=50):
    """Directional voting model (Rabinowitz-Macdonald).

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


short = "svdir"
alias = "svdir"
quote = "The spice must flow. -- Paul Atreides"
svdir = svdir


def cheatsheet() -> str:
    return "svdir({}) -> Directional voting model (Rabinowitz-Macdonald)."

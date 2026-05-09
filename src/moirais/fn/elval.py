# moirais.fn — function file (hadesllm/moirais)
"""Valence electoral model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elval(data=None, n=50):
    """Valence electoral model.

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


short = "elval"
alias = "elval"
quote = "The spice must flow. -- Paul Atreides"
elval = elval


def cheatsheet() -> str:
    return "elval({}) -> Valence electoral model."

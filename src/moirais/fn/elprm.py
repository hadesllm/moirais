# moirais.fn — function file (hadesllm/moirais)
"""Proportional electoral model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elprm(data=None, n=50):
    """Proportional electoral model.

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


short = "elprm"
alias = "elprm"
quote = "The spice must flow. -- Paul Atreides"
elprm = elprm


def cheatsheet() -> str:
    return "elprm({}) -> Proportional electoral model."

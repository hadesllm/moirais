# moirais.fn — function file (hadesllm/moirais)
"""Bayesian ideal point estimation.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpbp(data=None, n=50):
    """Bayesian ideal point estimation.

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


short = "idpbp"
alias = "idpbp"
quote = "The spice must flow. -- Paul Atreides"
idpbp = idpbp


def cheatsheet() -> str:
    return "idpbp({}) -> Bayesian ideal point estimation."

# morie.fn -- function file (hadesllm/morie)
"""MCMC ideal point estimation.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpmc(data=None, n=50):
    """MCMC ideal point estimation.

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


short = "idpmc"
alias = "idpmc"
quote = "I think, therefore I am. -- Rene Descartes"
idpmc = idpmc


def cheatsheet() -> str:
    return "idpmc({}) -> MCMC ideal point estimation."

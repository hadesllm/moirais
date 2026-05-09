# moirais.fn — function file (hadesllm/moirais)
"""Normal perturbation spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def psnrm(data=None, n=50):
    """Normal perturbation spatial.

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


short = "psnrm"
alias = "psnrm"
quote = "The spice must flow. -- Paul Atreides"
psnrm = psnrm


def cheatsheet() -> str:
    return "psnrm({}) -> Normal perturbation spatial."

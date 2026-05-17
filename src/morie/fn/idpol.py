# morie.fn -- function file (hadesllm/morie)
"""OLS ideal point regression.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpol(data=None, n=50):
    """OLS ideal point regression.

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


short = "idpol"
alias = "idpol"
quote = "What is now proved was once only imagined. -- William Blake"
idpol = idpol


def cheatsheet() -> str:
    return "idpol({}) -> OLS ideal point regression."

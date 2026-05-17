"""Constrained spatial utility max.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def umcon(data=None, n=50):
    """Constrained spatial utility max.

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


short = "umcon"
alias = "umcon"
quote = "Logic is the foundation of all certain knowledge. -- Leonhard Euler"
umcon = umcon


def cheatsheet() -> str:
    return "umcon({}) -> Constrained spatial utility max."

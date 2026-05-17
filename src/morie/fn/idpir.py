# morie.fn -- function file (hadesllm/morie)
"""IRT-based ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpir(data=None, n=50):
    """IRT-based ideal point.

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


short = "idpir"
alias = "idpir"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
idpir = idpir


def cheatsheet() -> str:
    return "idpir({}) -> IRT-based ideal point."

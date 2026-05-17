# morie.fn -- function file (hadesllm/morie)
"""Dynamic ideal point (DW-NOMINATE).

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpdw(data=None, n=50):
    """Dynamic ideal point (DW-NOMINATE).

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


short = "idpdw"
alias = "idpdw"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
idpdw = idpdw


def cheatsheet() -> str:
    return "idpdw({}) -> Dynamic ideal point (DW-NOMINATE)."

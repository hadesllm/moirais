# morie.fn -- function file (hadesllm/morie)
"""HMM spatial voting.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pshmm(data=None, n=50):
    """HMM spatial voting.

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


short = "pshmm"
alias = "pshmm"
quote = "We must know. We will know. -- David Hilbert"
pshmm = pshmm


def cheatsheet() -> str:
    return "pshmm({}) -> HMM spatial voting."

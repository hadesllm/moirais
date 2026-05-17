# morie.fn -- function file (hadesllm/morie)
"""Policy salience model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ispol(data=None, n=50):
    """Policy salience model.

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


short = "ispol"
alias = "ispol"
quote = "If I have seen further it is by standing on the shoulders of giants. -- Isaac Newton"
ispol = ispol


def cheatsheet() -> str:
    return "ispol({}) -> Policy salience model."

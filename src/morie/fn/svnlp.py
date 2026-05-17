"""Normal distribution vote probability.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svnlp(data=None, n=50):
    """Normal distribution vote probability.

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


short = "svnlp"
alias = "svnlp"
quote = "To understand God's thoughts we must study statistics. -- Florence Nightingale"
svnlp = svnlp


def cheatsheet() -> str:
    return "svnlp({}) -> Normal distribution vote probability."

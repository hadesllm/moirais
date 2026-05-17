"""
SVM species model

Category: WildlSp
"""

import numpy as np


def wlsvm(abundance=None, coords=None, n=50):
    """SVM species model

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if abundance is None:
        abundance = np.random.default_rng(0).poisson(10, n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(abundance))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(abundance), "total": int(np.sum(abundance)), "mean": float(np.mean(abundance))},
    )


short = "wlsvm"
alias = "wlsvm"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
wlsvm = wlsvm


def cheatsheet() -> str:
    return "wlsvm({}) -> SVM species model"

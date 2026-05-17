# morie.fn -- function file (hadesllm/morie)
"""
Growth rate spatial

Category: ForstSp
"""

import numpy as np


def fogrth(dbh=None, height=None, coords=None, n=50):
    """Growth rate spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if dbh is None:
        dbh = np.random.default_rng(0).uniform(5, 80, n)
    if height is None:
        height = np.random.default_rng(1).uniform(3, 40, n)
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    basal_area = np.pi * (dbh / 200) ** 2
    stat = float(np.sum(basal_area))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "n": len(dbh),
            "mean_dbh": float(np.mean(dbh)),
            "mean_height": float(np.mean(height)),
            "total_ba": float(np.sum(basal_area)),
        },
    )


short = "fogrth"
alias = "fogrth"
quote = "You have power over your mind, not outside events. -- Marcus Aurelius"
fogrth = fogrth


def cheatsheet() -> str:
    return "fogrth({}) -> Growth rate spatial"

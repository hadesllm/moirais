# morie.fn -- function file (hadesllm/morie)
"""
Height distribution spatial

Category: ForstSp
"""

import numpy as np


def foht(dbh=None, height=None, coords=None, n=50):
    """Height distribution spatial

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


short = "foht"
alias = "foht"
quote = "I'm gonna be King of the Pirates! -- Luffy"
foht = foht


def cheatsheet() -> str:
    return "foht({}) -> Height distribution spatial"

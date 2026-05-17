# morie.fn -- function file (hadesllm/morie)
"""
Local Geary's C

Category: SpatAutoC
"""

import numpy as np


def salge(values=None, w=None, n=50):
    """Local Geary's C

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if values is None:
        values = np.random.default_rng(0).standard_normal(n)
    if w is None:
        w = np.zeros((len(values), len(values)))
        for i in range(len(values)):
            for j in range(max(0, i - 1), min(len(values), i + 2)):
                if i != j:
                    w[i, j] = 1.0
        rs = w.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        w = w / rs
    n_ = len(values)
    mean_v = np.mean(values)
    dev = values - mean_v
    num = float(np.sum(w * np.outer(dev, dev)))
    den = float(np.sum(dev**2))
    stat = float(n_ * num / (np.sum(w) * den)) if den > 0 else 0.0
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": n_, "mean": float(mean_v), "numerator": num, "denominator": den},
    )


short = "salge"
alias = "salge"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
salge = salge


def cheatsheet() -> str:
    return "salge({}) -> Local Geary's C"

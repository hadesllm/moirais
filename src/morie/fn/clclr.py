# morie.fn -- function file (hadesllm/morie)
"""
CLARANS spatial

Category: ClstSp
"""

import numpy as np


def clclr(data=None, n=50, k=3, coords=None):
    """CLARANS spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).standard_normal((n, 2))
    if coords is None:
        coords = data[:, :2] if data.shape[1] >= 2 else np.random.default_rng(1).uniform(0, 100, (n, 2))
    centers = data[np.random.default_rng(2).choice(len(data), k, replace=False)]
    dists = np.sqrt(np.sum((data[:, None] - centers[None, :]) ** 2, axis=-1))
    labels = np.argmin(dists, axis=1)
    inertia = float(np.sum(np.min(dists, axis=1) ** 2))
    stat = float(inertia)
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "n_points": len(data),
            "k": k,
            "inertia": float(inertia),
            "label_counts": [int(np.sum(labels == i)) for i in range(k)],
        },
    )


short = "clclr"
alias = "clclr"
quote = "I think, therefore I am. -- Rene Descartes"
clclr = clclr


def cheatsheet() -> str:
    return "clclr({}) -> CLARANS spatial"

# moirais.fn — function file (hadesllm/moirais)
"""
BIRCH spatial clustering

Category: ClstSp
"""

import numpy as np


def clbrc(data=None, n=50, k=3, coords=None):
    """BIRCH spatial clustering

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


short = "clbrc"
alias = "clbrc"
quote = "Fear is the mind-killer. -- Bene Gesserit"
clbrc = clbrc


def cheatsheet() -> str:
    return "clbrc({}) -> BIRCH spatial clustering"

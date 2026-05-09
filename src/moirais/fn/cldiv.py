# moirais.fn — function file (hadesllm/moirais)
"""
Divisive spatial cluster

Category: ClstSp
"""

import numpy as np


def cldiv(data=None, n=50, k=3, coords=None):
    """Divisive spatial cluster

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


short = "cldiv"
alias = "cldiv"
quote = "Those who break the rules are scum. -- Kakashi"
cldiv = cldiv


def cheatsheet() -> str:
    return "cldiv({}) -> Divisive spatial cluster"

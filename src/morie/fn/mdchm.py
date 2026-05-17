# morie.fn -- function file (hadesllm/morie)
"""Chaos theorem multidim.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mdchm(data=None, query=None, origin=None, *, n=50):
    """Chaos theorem multidim.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal((n, 2))
    data = np.atleast_2d(np.asarray(data, dtype=float))
    if query is not None:
        query = np.asarray(query, dtype=float)
    if origin is not None:
        origin = np.asarray(origin, dtype=float)
    centroid = data.mean(axis=0)
    if query is not None and origin is not None:
        stat = float(np.linalg.norm(query - origin))
    else:
        stat = float(np.linalg.norm(centroid))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "centroid": centroid.tolist()},
    )


short = "mdchm"
alias = "mdchm"
quote = "What is now proved was once only imagined. -- William Blake"
mdchm = mdchm


def cheatsheet() -> str:
    return "mdchm({}) -> Chaos theorem multidim."

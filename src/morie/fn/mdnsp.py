# morie.fn -- function file (hadesllm/morie)
"""Non-separable preferences multidim.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mdnsp(data=None, query=None, origin=None, *, n=50):
    """Non-separable preferences multidim.

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


short = "mdnsp"
alias = "mdnsp"
quote = "The spice must flow. -- Paul Atreides"
mdnsp = mdnsp


def cheatsheet() -> str:
    return "mdnsp({}) -> Non-separable preferences multidim."

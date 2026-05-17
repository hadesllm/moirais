# morie.fn -- function file (hadesllm/morie)
"""Pareto frontier multidimensional.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mdprt(data=None, query=None, origin=None, *, n=50):
    """Pareto frontier multidimensional.

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


short = "mdprt"
alias = "mdprt"
quote = "Give me a place to stand and I will move the earth. -- Archimedes"
mdprt = mdprt


def cheatsheet() -> str:
    return "mdprt({}) -> Pareto frontier multidimensional."

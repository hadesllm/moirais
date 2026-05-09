# moirais.fn — function file (hadesllm/moirais)
"""Cycling multidimensional voting.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mdcyc(data=None, query=None, origin=None, *, n=50):
    """Cycling multidimensional voting.

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


short = "mdcyc"
alias = "mdcyc"
quote = "The spice must flow. -- Paul Atreides"
mdcyc = mdcyc


def cheatsheet() -> str:
    return "mdcyc({}) -> Cycling multidimensional voting."

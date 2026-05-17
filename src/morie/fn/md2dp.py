# morie.fn -- function file (hadesllm/morie)
"""2D spatial position equilibrium.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def md2dp(data=None, query=None, origin=None, *, n=50):
    """2D spatial position equilibrium.

    Parameters
    ----------
    data : array-like, optional
        2D point coordinates (n, 2).
    query : array-like, optional
        Query point coordinates.
    origin : array-like, optional
        Origin point coordinates.

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


short = "md2dp"
alias = "md2dp"
quote = "Knowledge is power. -- Francis Bacon"
md2dp = md2dp


def cheatsheet() -> str:
    return "md2dp({}) -> 2D spatial position equilibrium."

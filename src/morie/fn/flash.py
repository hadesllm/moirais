# morie.fn -- function file (hadesllm/morie)
"""A journey of a thousand miles begins with a single step. -- Lao Tzu"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def fast_ann(
    data: np.ndarray | pd.DataFrame,
    query: np.ndarray | None = None,
    *,
    k: int = 5,
    n_trees: int = 10,
    seed: int | None = None,
) -> DescriptiveResult:
    """Approximate nearest neighbor search via random projection trees.

    Builds *n_trees* random projection trees and queries them for the *k*
    nearest neighbors.  Much faster than brute-force for high-dimensional data.

    Parameters
    ----------
    data : array (n, d)
        Reference points.
    query : array (m, d) or None
        Query points.  If None, uses the first point in *data*.
    k : int
        Number of neighbors to return.
    n_trees : int
        Number of random projection trees (more = better recall).
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` = dict with ``'indices'`` and ``'distances'`` arrays.
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy(dtype=float)
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, d = data.shape
    if query is None:
        query = data[:1]
    query = np.asarray(query, dtype=float)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    candidates: set[int] = set()
    for _ in range(n_trees):
        proj = rng.standard_normal(d)
        proj /= np.linalg.norm(proj)
        scores = data @ proj
        q_score = query[0] @ proj
        order = np.argsort(scores)
        idx = np.searchsorted(scores[order], q_score)
        lo = max(0, idx - k)
        hi = min(n, idx + k)
        candidates.update(order[lo:hi].tolist())
    cand_arr = np.array(list(candidates))
    dists = np.linalg.norm(data[cand_arr] - query[0], axis=1)
    top_k = np.argsort(dists)[:k]
    indices = cand_arr[top_k]
    distances = dists[top_k]
    return DescriptiveResult(
        name="Fast approximate nearest neighbors",
        value={"indices": indices.tolist(), "distances": distances.tolist()},
        extra={"k": k, "n": n, "d": d, "n_trees": n_trees, "n_candidates": len(candidates)},
    )


flash = fast_ann


def cheatsheet() -> str:
    return "fast_ann({}) -> Fast approximate nearest neighbor search. 'My name is Barry "

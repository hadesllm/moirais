# morie.fn — function file (hadesllm/morie)
"""Convex hull per group in MDS space. 'Cursed Energy.' -- Gojo, Jujutsu Kaisen"""

from __future__ import annotations

from ._containers import DescriptiveResult


def convex_hull_mds(X, groups):
    """Compute convex hull vertices per group in 2D MDS space.

    Parameters
    ----------
    X : array-like
        Coordinate matrix (n x 2).
    groups : array-like
        Group labels (length n).

    Returns
    -------
    DescriptiveResult
        value = dict of group -> hull vertex indices.
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    hulls = {}
    for g in unique:
        mask = groups == g
        pts = X[mask]
        if len(pts) < 3:
            hulls[str(g)] = list(range(len(pts)))
            continue
        cx, cy = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        order = np.argsort(angles)
        hulls[str(g)] = pts[order].tolist()
    return DescriptiveResult(name="convex_hull_mds", value=hulls, extra={"n_groups": len(unique)})


mdsex = convex_hull_mds


def cheatsheet() -> str:
    return "convex_hull_mds({}) -> Convex hull per group in MDS space. 'Cursed Energy.' -- Gojo"

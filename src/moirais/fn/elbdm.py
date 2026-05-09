# moirais.fn — function file (hadesllm/moirais)
"""Elbow method for MDS dimensionality. 'Malevolent Shrine.' -- Sukuna, Jujutsu Kaisen"""

from __future__ import annotations

from ._containers import DescriptiveResult


def elbow_mds_dim(stress_values):
    """Find optimal MDS dimensionality via elbow/knee detection.

    Parameters
    ----------
    stress_values : list or array-like
        Stress values for dims 1, 2, ..., len(stress_values).

    Returns
    -------
    DescriptiveResult
        value = optimal dimension count (int).
    """
    import numpy as np

    sv = np.asarray(stress_values, dtype=float)
    n = len(sv)
    if n <= 2:
        return DescriptiveResult(name="elbow_mds_dim", value=1, extra={"stress_values": sv.tolist()})

    dims = np.arange(1, n + 1, dtype=float)
    p1 = np.array([dims[0], sv[0]])
    p2 = np.array([dims[-1], sv[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    best_dist = -1.0
    best_dim = 1
    for i in range(n):
        pt = np.array([dims[i], sv[i]])
        v = p1 - pt
        dist = abs(line_vec[0] * v[1] - line_vec[1] * v[0]) / max(line_len, 1e-12)
        if dist > best_dist:
            best_dist = dist
            best_dim = int(dims[i])
    return DescriptiveResult(name="elbow_mds_dim", value=best_dim, extra={"stress_values": sv.tolist()})


elbdm = elbow_mds_dim


def cheatsheet() -> str:
    return "elbow_mds_dim({}) -> Elbow method for MDS dimensionality. 'Malevolent Shrine.' --"

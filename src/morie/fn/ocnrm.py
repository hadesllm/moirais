# morie.fn — function file (hadesllm/morie)
"""OC normal vector for a single vote."""

from __future__ import annotations

from ._containers import DescriptiveResult


def oc_normal_vector(X, votes_j) -> DescriptiveResult:
    """Compute normal vector for vote j via mean-difference.

    .. epigraph:: "Run." -- Walter White, Breaking Bad
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    v = np.asarray(votes_j, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    valid = ~np.isnan(v)
    X_v = X[valid]
    v_v = v[valid]
    m_yea = np.mean(X_v[v_v == 1], axis=0) if np.any(v_v == 1) else np.zeros(X.shape[1])
    m_nay = np.mean(X_v[v_v == 0], axis=0) if np.any(v_v == 0) else np.zeros(X.shape[1])
    normal = m_yea - m_nay
    norm_len = np.linalg.norm(normal)
    if norm_len > 1e-15:
        normal = normal / norm_len
    return DescriptiveResult(
        name="oc_normal_vector",
        value=float(norm_len),
        extra={
            "normal": normal.tolist(),
            "norm": float(norm_len),
            "n_yea": int(np.sum(v_v == 1)),
            "n_nay": int(np.sum(v_v == 0)),
        },
    )


ocnrm = oc_normal_vector


def cheatsheet() -> str:
    return "oc_normal_vector({}) -> OC normal vector for a single vote."

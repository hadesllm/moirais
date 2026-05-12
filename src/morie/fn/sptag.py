# morie.fn -- function file (hadesllm/morie)
"""Spatial agreement scores between legislators (Armstrong Ch 8)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatial_agreement", "sptag"]


def spatial_agreement(x):
    """Pairwise agreement matrix: A_ij = proportion of votes on which
    legislators i and j voted the same way (excluding absences).

    Parameters
    ----------
    x : (n, m) binary vote matrix; NaN = absent.
        A 1-D vector of length n is interpreted as a single roll-call
        column (degenerate but supported).

    Returns
    -------
    RichResult with keys: agreement, mean_agreement, n, m
    """
    M = np.asarray(x, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    n, m = M.shape
    if n < 2:
        return RichResult(payload={"agreement": np.eye(n),
                                   "mean_agreement": np.nan, "n": n,
                                   "m": m, "method": "spatial_agreement"})
    A = np.eye(n)
    valid = ~np.isnan(M)
    for i in range(n):
        for j in range(i + 1, n):
            both = valid[i] & valid[j]
            denom = float(np.sum(both))
            if denom == 0:
                A[i, j] = A[j, i] = np.nan
            else:
                same = float(np.sum(M[i][both] == M[j][both]))
                A[i, j] = A[j, i] = same / denom
    # Mean off-diagonal agreement
    iu = np.triu_indices(n, k=1)
    off = A[iu]
    mean_a = float(np.nanmean(off)) if off.size else np.nan
    return RichResult(
        title="Pairwise vote agreement (Armstrong Ch 8)",
        summary_lines=[("Mean off-diagonal agreement", mean_a),
                       ("n legislators", n), ("m votes", m)],
        payload={"agreement": A, "mean_agreement": mean_a,
                 "n": int(n), "m": int(m),
                 "method": "spatial_agreement"},
    )


sptag = spatial_agreement


def cheatsheet():
    return "sptag: Pairwise agreement matrix A_ij = P(same vote)."


# CANONICAL TEST
# >>> M = np.array([[1,1,0],[1,1,0],[0,0,1]], dtype=float)
# >>> r = spatial_agreement(M)
# >>> assert r["agreement"][0,1] == 1.0

# morie.fn -- function file (hadesllm/morie)
"""OC (optimal classification) scaling."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def oc_scaling(vote_matrix, n_dims: int = 1) -> DescriptiveResult:
    """Optimal classification (Poole 2000) nonparametric scaling.

    Finds legislator ideal points that maximise correctly classified votes
    via cutting lines (hyperplanes). Uses SVD of the agreement matrix as
    a starting solution.

    :param vote_matrix: (n_legislators x n_votes) binary matrix (1=yea, 0=nay, NaN=missing).
    :param n_dims: Number of dimensions (1 or 2).
    :return: DescriptiveResult with ideal points and classification success.

    References
    ----------
    Armstrong (2014), Ch 7. Poole (2000).

    .. epigraph:: Mathematics is the queen of the sciences. -- Carl Friedrich Gauss
    """
    V = np.asarray(vote_matrix, dtype=float)
    if V.ndim != 2:
        raise ValueError("vote_matrix must be 2D.")
    n_leg, n_votes = V.shape

    agree = np.zeros((n_leg, n_leg))
    for i in range(n_leg):
        for j in range(i + 1, n_leg):
            mask = ~np.isnan(V[i]) & ~np.isnan(V[j])
            if mask.sum() > 0:
                agreement = np.mean(V[i, mask] == V[j, mask])
                agree[i, j] = agree[j, i] = agreement

    U, S, Vt = np.linalg.svd(agree)
    ideal_points = U[:, :n_dims] * S[:n_dims]

    correct = 0
    total = 0
    for j in range(n_votes):
        valid = ~np.isnan(V[:, j])
        if valid.sum() < 2:
            continue
        yea = ideal_points[valid & (V[:, j] == 1)]
        nay = ideal_points[valid & (V[:, j] == 0)]
        if len(yea) == 0 or len(nay) == 0:
            continue
        yea_mean = yea.mean(axis=0)
        nay_mean = nay.mean(axis=0)
        midpoint = (yea_mean + nay_mean) / 2
        normal = yea_mean - nay_mean
        pts = ideal_points[valid]
        proj = (pts - midpoint) @ normal.reshape(-1, 1)
        pred = (proj.ravel() > 0).astype(float)
        correct += int((pred == V[valid, j]).sum())
        total += int(valid.sum())

    clf_rate = correct / max(total, 1)
    return DescriptiveResult(
        name="oc_scaling",
        value={"ideal_points": ideal_points, "classification_rate": clf_rate},
        extra={"n_legislators": n_leg, "n_votes": n_votes, "n_dims": n_dims},
    )


ocslt = oc_scaling


def cheatsheet() -> str:
    return "oc_scaling({}) -> Optimal classification (OC) scaling."

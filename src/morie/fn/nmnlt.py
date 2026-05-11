# morie.fn — function file (hadesllm/morie)
"""NOMINATE scaling (1D/2D) for legislative ideal points."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def nominate_scaling(
    vote_matrix,
    n_dims: int = 1,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    beta: float = 15.0,
    w: float = 0.5,
) -> DescriptiveResult:
    """NOMINATE-style maximum likelihood estimation of ideal points.

    Estimates legislator ideal points and bill midpoints by maximising
    the likelihood of observed votes under a Gaussian spatial utility
    model. Simplified version of Poole & Rosenthal (1997).

    :param vote_matrix: (n_legislators x n_votes) binary matrix.
    :param n_dims: 1 or 2 dimensions.
    :param max_iter: Maximum EM iterations.
    :param tol: Convergence tolerance on log-likelihood.
    :param beta: Spatial utility precision.
    :param w: Signal-to-noise ratio parameter.
    :return: DescriptiveResult with ideal points and bill midpoints.

    References
    ----------
    Armstrong (2014), Ch 8. Poole & Rosenthal (1997).

    .. epigraph:: "The ability to speak does not make you intelligent." -- Qui-Gon, Star Wars
    """
    V = np.asarray(vote_matrix, dtype=float)
    if V.ndim != 2:
        raise ValueError("vote_matrix must be 2D.")
    n_leg, n_votes = V.shape

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_leg, n_dims)) * 0.1
    Z_yea = rng.standard_normal((n_votes, n_dims)) * 0.1
    Z_nay = rng.standard_normal((n_votes, n_dims)) * 0.1

    def _loglik(X, Z_yea, Z_nay):
        ll = 0.0
        for j in range(n_votes):
            valid = ~np.isnan(V[:, j])
            d_yea = np.sum((X[valid] - Z_yea[j]) ** 2, axis=1)
            d_nay = np.sum((X[valid] - Z_nay[j]) ** 2, axis=1)
            u_yea = -beta * d_yea
            u_nay = -beta * d_nay
            exp_diff = np.clip(w * (u_yea - u_nay), -500, 500)
            prob_yea = 1.0 / (1.0 + np.exp(-exp_diff))
            prob_yea = np.clip(prob_yea, 1e-10, 1 - 1e-10)
            votes = V[valid, j]
            ll += (votes * np.log(prob_yea) + (1 - votes) * np.log(1 - prob_yea)).sum()
        return ll

    prev_ll = _loglik(X, Z_yea, Z_nay)
    for _ in range(max_iter):
        for j in range(n_votes):
            valid = ~np.isnan(V[:, j])
            yea_pts = X[valid & (V[:, j] == 1)]
            nay_pts = X[valid & (V[:, j] == 0)]
            if len(yea_pts) > 0:
                Z_yea[j] = yea_pts.mean(axis=0)
            if len(nay_pts) > 0:
                Z_nay[j] = nay_pts.mean(axis=0)

        for i in range(n_leg):
            valid = ~np.isnan(V[i])
            voted_yea = valid & (V[i] == 1)
            voted_nay = valid & (V[i] == 0)
            pos = []
            if voted_yea.sum() > 0:
                pos.append(Z_yea[voted_yea].mean(axis=0))
            if voted_nay.sum() > 0:
                pos.append(Z_nay[voted_nay].mean(axis=0))
            if pos:
                X[i] = np.mean(pos, axis=0)

        cur_ll = _loglik(X, Z_yea, Z_nay)
        if abs(cur_ll - prev_ll) < tol:
            break
        prev_ll = cur_ll

    return DescriptiveResult(
        name="nominate_scaling",
        value={"ideal_points": X, "yea_points": Z_yea, "nay_points": Z_nay},
        extra={"log_likelihood": float(prev_ll), "n_dims": n_dims, "beta": beta},
    )


nmnlt = nominate_scaling


def cheatsheet() -> str:
    return "nominate_scaling({}) -> NOMINATE scaling for ideal points."

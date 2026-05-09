# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian ideal point estimation via MCMC."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bayesian_ideal_points(
    vote_matrix,
    n_dims: int = 1,
    *,
    n_iter: int = 1000,
    burnin: int = 200,
    beta: float = 1.0,
    prior_sd: float = 1.0,
    seed: int = 42,
) -> DescriptiveResult:
    """Bayesian ideal point estimation via Metropolis-Hastings MCMC.

    Places normal priors on ideal points and bill parameters, then
    samples from the posterior using a random-walk MH algorithm.

    :param vote_matrix: (n_legislators x n_votes) binary matrix.
    :param n_dims: Dimensionality of ideal point space.
    :param n_iter: MCMC iterations (post-burnin).
    :param burnin: Burnin iterations to discard.
    :param beta: Spatial precision parameter.
    :param prior_sd: Prior standard deviation for ideal points.
    :param seed: Random seed.
    :return: DescriptiveResult with posterior means and credible intervals.

    References
    ----------
    Armstrong (2014), Ch 9. Clinton, Jackman & Rivers (2004).

    .. epigraph:: "Truly wonderful the mind of a child is.", Star Wars
    """
    V = np.asarray(vote_matrix, dtype=float)
    if V.ndim != 2:
        raise ValueError("vote_matrix must be 2D.")
    n_leg, n_votes = V.shape
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n_leg, n_dims)) * 0.5
    Z_yea = rng.standard_normal((n_votes, n_dims)) * 0.5
    Z_nay = rng.standard_normal((n_votes, n_dims)) * 0.5

    def _logpost(X, Z_yea, Z_nay):
        lp = -0.5 * np.sum(X ** 2) / (prior_sd ** 2)
        lp -= 0.5 * np.sum(Z_yea ** 2) / (prior_sd ** 2)
        lp -= 0.5 * np.sum(Z_nay ** 2) / (prior_sd ** 2)
        for j in range(n_votes):
            valid = ~np.isnan(V[:, j])
            d_yea = np.sum((X[valid] - Z_yea[j]) ** 2, axis=1)
            d_nay = np.sum((X[valid] - Z_nay[j]) ** 2, axis=1)
            logit = np.clip(beta * (d_nay - d_yea), -500, 500)
            prob = 1.0 / (1.0 + np.exp(-logit))
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            votes = V[valid, j]
            lp += (votes * np.log(prob) + (1 - votes) * np.log(1 - prob)).sum()
        return lp

    samples = np.zeros((n_iter, n_leg, n_dims))
    prop_sd = 0.1
    cur_lp = _logpost(X, Z_yea, Z_nay)
    accept = 0

    for it in range(burnin + n_iter):
        for i in range(n_leg):
            X_prop = X.copy()
            X_prop[i] += rng.standard_normal(n_dims) * prop_sd
            prop_lp = _logpost(X_prop, Z_yea, Z_nay)
            if np.log(rng.random()) < prop_lp - cur_lp:
                X = X_prop
                cur_lp = prop_lp
                if it >= burnin:
                    accept += 1
        if it >= burnin:
            samples[it - burnin] = X

    post_mean = samples.mean(axis=0)
    post_sd = samples.std(axis=0)
    ci_lo = np.percentile(samples, 2.5, axis=0)
    ci_hi = np.percentile(samples, 97.5, axis=0)
    accept_rate = accept / max((n_iter * n_leg), 1)

    return DescriptiveResult(
        name="bayesian_ideal_points",
        value={
            "posterior_mean": post_mean,
            "posterior_sd": post_sd,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        },
        extra={"acceptance_rate": accept_rate, "n_iter": n_iter, "burnin": burnin},
    )


bayid = bayesian_ideal_points


def cheatsheet() -> str:
    return "bayesian_ideal_points({}) -> Bayesian ideal point estimation (MCMC)."

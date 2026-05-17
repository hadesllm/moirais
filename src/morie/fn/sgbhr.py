"""Bayesian hierarchical spatial model (Gibbs sampler)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def bayesian_hierarchical_spatial(
    Z: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    n_samples: int = 500,
    cov_params: dict | None = None,
    seed: int = 42,
) -> SpatialResult:
    r"""Bayesian hierarchical spatial model via Gibbs sampling.

    Two-level hierarchy:
    :math:`Z | \phi \sim N(X\beta + \phi, \tau^2 I)` and
    :math:`\phi \sim N(0, \sigma^2 R)`.

    Parameters
    ----------
    Z : np.ndarray
        Response, shape ``(n,)``.
    X : np.ndarray
        Design matrix, shape ``(n, p)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    n_samples : int
        MCMC samples.
    cov_params : dict, optional
        ``{"range": float}``.
    seed : int
        RNG seed.

    Returns
    -------
    SpatialResult
        ``statistic`` is posterior mean of beta[0].
        ``extra`` has ``beta_samples``, ``sigma2_samples``, ``tau2_samples``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

    """
    rng_obj = np.random.default_rng(seed)
    Z = np.asarray(Z, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    n, p = X.shape
    params = cov_params or {"range": 1.0}
    r = params.get("range", 1.0)

    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    R = np.exp(-dist / r)

    beta = np.zeros(p)
    sigma2 = 1.0
    tau2 = 1.0
    phi = np.zeros(n)

    beta_samples = np.empty((n_samples, p))
    sigma2_samples = np.empty(n_samples)
    tau2_samples = np.empty(n_samples)

    for s in range(n_samples):
        V_beta = np.linalg.inv(X.T @ X / tau2 + np.eye(p) * 0.01)
        m_beta = V_beta @ (X.T @ (Z - phi) / tau2)
        beta = rng_obj.multivariate_normal(m_beta, V_beta)

        resid = Z - X @ beta - phi
        a_tau = 0.01 + n / 2
        b_tau = 0.01 + np.sum(resid**2) / 2
        tau2 = 1.0 / rng_obj.gamma(a_tau, 1.0 / b_tau)

        R_inv = np.linalg.inv(sigma2 * R + 1e-6 * np.eye(n))
        V_phi = np.linalg.inv(R_inv + np.eye(n) / tau2)
        m_phi = V_phi @ ((Z - X @ beta) / tau2)
        phi = rng_obj.multivariate_normal(m_phi, V_phi)

        a_sig = 0.01 + n / 2
        b_sig = 0.01 + phi @ np.linalg.solve(R + 1e-6 * np.eye(n), phi) / 2
        sigma2 = 1.0 / rng_obj.gamma(a_sig, 1.0 / b_sig)

        beta_samples[s] = beta
        sigma2_samples[s] = sigma2
        tau2_samples[s] = tau2

    return SpatialResult(
        name="bayesian_hierarchical_spatial",
        statistic=float(np.mean(beta_samples[:, 0])),
        p_value=None,
        extra={
            "beta_samples": beta_samples,
            "sigma2_samples": sigma2_samples,
            "tau2_samples": tau2_samples,
        },
    )


sgbhr = bayesian_hierarchical_spatial


def cheatsheet() -> str:
    return "bayesian_hierarchical_spatial({}) -> Bayesian hierarchical spatial model (Gibbs sampler)."

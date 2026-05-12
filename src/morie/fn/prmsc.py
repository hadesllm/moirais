# morie.fn -- function file (hadesllm/morie)
"""All models are wrong, but some are useful. -- George E. P. Box"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def vae_sample(
    mu: np.ndarray,
    log_var: np.ndarray,
    *,
    n_samples: int = 1,
    seed: int = 42,
) -> DescriptiveResult:
    """Sample from a VAE latent space using the reparameterization trick.

    z = mu + sigma * epsilon, where epsilon ~ N(0, I)

    Parameters
    ----------
    mu : array-like
        Mean of the latent distribution (batch_size, latent_dim).
    log_var : array-like
        Log-variance of the latent distribution.
    n_samples : int
        Number of samples per input.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        With ``value`` = sampled latent vectors and ``extra`` containing KL divergence.
    """
    mu = np.asarray(mu, dtype=float)
    log_var = np.asarray(log_var, dtype=float)
    if mu.shape != log_var.shape:
        raise ValueError("mu and log_var must have same shape")

    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
        log_var = log_var.reshape(1, -1)

    batch, latent_dim = mu.shape
    rng = np.random.default_rng(seed)

    sigma = np.exp(0.5 * log_var)

    samples = np.zeros((batch, n_samples, latent_dim))
    for s in range(n_samples):
        eps = rng.standard_normal((batch, latent_dim))
        samples[:, s, :] = mu + sigma * eps

    kl_div = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1)
    mean_kl = float(kl_div.mean())

    if n_samples == 1:
        samples = samples.squeeze(axis=1)

    return DescriptiveResult(
        name="vae_sample",
        value=samples,
        extra={
            "kl_divergence": mean_kl,
            "per_sample_kl": kl_div,
            "latent_dim": latent_dim,
            "batch_size": batch,
            "n_samples": n_samples,
        },
    )


prmsc = vae_sample


def cheatsheet() -> str:
    return "All models are wrong, but some are useful. -- George E. P. Box"

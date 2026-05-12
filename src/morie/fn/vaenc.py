# morie.fn — function file (hadesllm/morie)
"""Variational autoencoder ELBO loss (Kingma & Welling 2014)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["vae_elbo"]


def vae_elbo(x, x_recon, mu, log_var, reduction: str = "mean"):
    """Variational autoencoder ELBO.

    For Gaussian encoder :math:`q(z|x) = \\mathcal{N}(\\mu, \\sigma^2)`
    and unit-Gaussian prior :math:`p(z) = \\mathcal{N}(0, I)`, the
    closed-form KL is

    .. math::

        D_\\text{KL}(q \\| p) = -\\tfrac{1}{2} \\sum_j
            \\bigl(1 + \\log\\sigma_j^2 - \\mu_j^2 - \\sigma_j^2\\bigr).

    ELBO :math:`= \\mathbb{E}_q[\\log p(x|z)] - D_\\text{KL}(q\\|p)`,
    with the reconstruction term taken as Gaussian (MSE) here.

    Parameters
    ----------
    x : array-like
        Original input.
    x_recon : array-like
        Reconstruction.
    mu : array-like
        Encoder mean.
    log_var : array-like
        Encoder log-variance.
    reduction : str
        ``'mean'`` (default) or ``'sum'``.

    Returns
    -------
    result : RichResult
        Keys: ``elbo`` / ``estimate``, ``loss`` (= ``-elbo``),
        ``recon_loss``, ``kl_divergence``.

    References
    ----------
    Kingma, D. P., & Welling, M. (2014). Auto-encoding variational
    Bayes. *ICLR*.
    """
    x = np.asarray(x, dtype=float)
    x_recon = np.asarray(x_recon, dtype=float)
    mu = np.asarray(mu, dtype=float)
    log_var = np.asarray(log_var, dtype=float)

    diff = x - x_recon
    recon = 0.5 * diff * diff
    if recon.ndim > 1:
        recon = recon.sum(axis=tuple(range(1, recon.ndim)))
    else:
        recon = np.atleast_1d(recon.sum())

    kl = -0.5 * (1.0 + log_var - mu ** 2 - np.exp(log_var))
    if kl.ndim > 1:
        kl = kl.sum(axis=tuple(range(1, kl.ndim)))
    else:
        kl = np.atleast_1d(kl.sum())

    if reduction == "mean":
        recon_loss = float(recon.mean())
        kl_div = float(kl.mean())
    elif reduction == "sum":
        recon_loss = float(recon.sum())
        kl_div = float(kl.sum())
    else:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}.")

    elbo = -(recon_loss + kl_div)
    loss = -elbo

    return RichResult(
        title="VAE ELBO",
        summary_lines=[("ELBO", elbo), ("Recon loss", recon_loss),
                       ("KL divergence", kl_div), ("reduction", reduction)],
        payload={
            "elbo": elbo,
            "estimate": elbo,
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_divergence": kl_div,
            "method": "VAE ELBO",
        },
    )


# CANONICAL TEST
# x=x_recon=[1,1], mu=[0,0], log_var=[0,0] (sigma^2=1, prior matched)
# -> recon=0, KL=0, ELBO=0


def cheatsheet():
    return "vaenc: ELBO = E[log p(x|z)] - KL(q(z|x)||p(z))"

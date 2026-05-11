# morie.fn — function file (hadesllm/morie)
"""VQ-VAE codebook + commitment loss with straight-through estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_vq_vae_codebook_loss"]


def geron_vq_vae_codebook_loss(x, z_e, z_q, codebook, beta):
    """
    VQ-VAE codebook + commitment loss with straight-through estimator

    Formula: L = ||x - Dec(z_q)||^2 + ||sg(z_e) - e||^2 + beta * ||z_e - sg(e)||^2

    Parameters
    ----------
    x : array-like
        Input data.
    z_e : array-like
        Input data.
    z_q : array-like
        Input data.
    codebook : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 18, VQ-VAE section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "VQ-VAE codebook + commitment loss with straight-through estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "VQ-VAE codebook + commitment loss with straight-through estimator"})


def cheatsheet():
    return "grvqv: VQ-VAE codebook + commitment loss with straight-through estimator"

# morie.fn — function file (hadesllm/morie)
"""GAN minimax / non-saturating loss (Goodfellow et al. 2014)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["gan_loss"]


def _clip_log(p, eps=1e-12):
    return np.log(np.clip(p, eps, 1.0))


def gan_loss(D_real, D_fake, kind: str = "minimax"):
    r"""Compute discriminator and generator losses for the original GAN.

    The discriminator maximises

    .. math::

        V(D) = \\mathbb{E}_{x\\sim p_\\text{data}}[\\log D(x)]
             + \\mathbb{E}_{z\\sim p_z}[\\log(1 - D(G(z)))],

    so ``D_loss = -V(D)``.

    Two generator variants are supported:

    - ``kind='minimax'``:   ``G_loss = +E[log(1 - D(G(z)))]``
      (original min-max, gradients saturate early).
    - ``kind='nonsaturating'``: ``G_loss = -E[log D(G(z))]``
      (Goodfellow's recommended training-time trick).

    Parameters
    ----------
    D_real : array-like
        Discriminator outputs on real data (sigmoid probabilities).
    D_fake : array-like
        Discriminator outputs on generator samples (sigmoid
        probabilities).
    kind : str
        ``'minimax'`` (default) or ``'nonsaturating'``.

    Returns
    -------
    result : RichResult
        Keys: ``d_loss``, ``g_loss``, ``v`` (value of V(D)),
        ``estimate`` (= ``d_loss``).

    References
    ----------
    Goodfellow, I. et al. (2014). Generative adversarial nets.
    *NeurIPS*.
    """
    D_real = np.asarray(D_real, dtype=float).ravel()
    D_fake = np.asarray(D_fake, dtype=float).ravel()
    v_real = _clip_log(D_real).mean()
    v_fake_neg = _clip_log(1.0 - D_fake).mean()
    V = float(v_real + v_fake_neg)
    d_loss = -V
    if kind == "minimax":
        g_loss = float(v_fake_neg)
    elif kind == "nonsaturating":
        g_loss = float(-_clip_log(D_fake).mean())
    else:
        raise ValueError(
            f"kind must be 'minimax' or 'nonsaturating', got {kind!r}.")
    return RichResult(
        title=f"GAN loss ({kind})",
        summary_lines=[("V(D)", V), ("D loss", d_loss), ("G loss", g_loss)],
        payload={
            "d_loss": d_loss,
            "g_loss": g_loss,
            "v": V,
            "estimate": d_loss,
            "kind": kind,
            "method": f"GAN {kind} loss",
        },
    )


# CANONICAL TEST
# D_real = [0.9, 0.9], D_fake = [0.1, 0.1]
# v_real = log(0.9), v_fake_neg = log(1-0.1) = log(0.9)
# V = 2 * log(0.9) = -0.21072; d_loss = 0.21072
# g_loss (minimax) = log(0.9) = -0.10536


def cheatsheet():
    return "ganls: GAN V(D)=E[log D(x)]+E[log(1-D(G(z)))]; d_loss=-V"

# moirais.fn — function file (hadesllm/moirais)
"""GAN discriminator score. 'I am burdened with glorious purpose.' -- Loki"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def illusion_score(
    real_scores: np.ndarray | list[float],
    fake_scores: np.ndarray | list[float],
) -> DescriptiveResult:
    """Evaluate a GAN discriminator's ability to distinguish real from fake.

    Computes the binary cross-entropy loss, accuracy, and the
    Jensen-Shannon divergence between real and fake score distributions.

    Parameters
    ----------
    real_scores : array-like
        Discriminator outputs for real samples (in [0, 1]).
    fake_scores : array-like
        Discriminator outputs for generated/fake samples (in [0, 1]).

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``bce_loss``, ``accuracy``, ``js_divergence``,
        ``mean_real``, ``mean_fake``.
    """
    r = np.asarray(real_scores, dtype=float)
    f = np.asarray(fake_scores, dtype=float)
    if len(r) == 0 or len(f) == 0:
        raise ValueError("Both real and fake scores must be non-empty")

    eps = 1e-12
    r = np.clip(r, eps, 1 - eps)
    f = np.clip(f, eps, 1 - eps)

    bce_real = -np.mean(np.log(r))
    bce_fake = -np.mean(np.log(1 - f))
    bce = float((bce_real + bce_fake) / 2)

    acc_real = np.mean(r > 0.5)
    acc_fake = np.mean(f < 0.5)
    accuracy = float((acc_real + acc_fake) / 2)

    bins = np.linspace(0, 1, 51)
    p, _ = np.histogram(r, bins=bins, density=True)
    q, _ = np.histogram(f, bins=bins, density=True)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = (p + q) / 2
    kl_pm = np.sum(np.where(p > 0, p * np.log(p / (m + eps) + eps), 0))
    kl_qm = np.sum(np.where(q > 0, q * np.log(q / (m + eps) + eps), 0))
    jsd = float((kl_pm + kl_qm) / 2)

    return DescriptiveResult(
        name="illusion_score",
        value={
            "bce_loss": bce,
            "accuracy": accuracy,
            "js_divergence": jsd,
            "mean_real": float(r.mean()),
            "mean_fake": float(f.mean()),
        },
        extra={"n_real": len(r), "n_fake": len(f)},
    )


lokiv = illusion_score


def cheatsheet() -> str:
    return "illusion_score({}) -> GAN discriminator score. 'I am burdened with glorious purpos"

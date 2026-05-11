"""Adversarial perturbation (FGSM-style). 'Deception is an art.' -- Sideways"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def adversarial_perturb(
    x: np.ndarray,
    gradient: np.ndarray,
    *,
    epsilon: float = 0.1,
    method: str = "fgsm",
    seed: int = 42,
) -> DescriptiveResult:
    """Generate adversarial perturbation for a data point.

    Implements FGSM (Goodfellow et al., 2015) and random perturbation.

    Parameters
    ----------
    x : array-like
        Original input.
    gradient : array-like
        Gradient of the loss w.r.t. x.
    epsilon : float
        Perturbation magnitude.
    method : str
        'fgsm' (fast gradient sign) or 'pgd_step' (projected gradient).
    seed : int
        Random seed (for any stochastic component).

    Returns
    -------
    DescriptiveResult
        With ``value`` = perturbed input and ``extra`` containing
        perturbation norm.
    """
    x = np.asarray(x, dtype=float)
    g = np.asarray(gradient, dtype=float)
    if x.shape != g.shape:
        raise ValueError("x and gradient must have same shape")

    if method == "fgsm":
        perturbation = epsilon * np.sign(g)
    elif method == "pgd_step":
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-30:
            perturbation = epsilon * g / g_norm
        else:
            perturbation = np.zeros_like(g)
    else:
        raise ValueError(f"Unknown method: {method}")

    x_adv = x + perturbation
    l2_dist = float(np.linalg.norm(perturbation))
    linf_dist = float(np.max(np.abs(perturbation)))

    return DescriptiveResult(
        name="adversarial_perturb",
        value=x_adv,
        extra={"method": method, "epsilon": epsilon, "l2_norm": l2_dist, "linf_norm": linf_dist},
    )


sidwy = adversarial_perturb


def cheatsheet() -> str:
    return "adversarial_perturb({}) -> Adversarial perturbation (FGSM-style). 'Deception is an art."

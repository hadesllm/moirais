"""Simulate 2D unfolding data. 'Ultra Instinct.' -- Goku, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def simulate_2d_unfolding(n_resp=30, n_stim=5, noise=0.1, seed=42):
    """Generate synthetic 2D unfolding data (ideal points + stimuli + preferences).

    Parameters
    ----------
    n_resp : int
        Number of respondents.
    n_stim : int
        Number of stimuli.
    noise : float
        Gaussian noise std added to distances.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        value = observed distance matrix (n_resp x n_stim).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    X_resp = rng.standard_normal((n_resp, 2))
    X_stim = rng.standard_normal((n_stim, 2)) * 0.5
    D = np.zeros((n_resp, n_stim))
    for i in range(n_resp):
        for j in range(n_stim):
            D[i, j] = np.sqrt(np.sum((X_resp[i] - X_stim[j]) ** 2))
    D += rng.normal(0, noise, D.shape)
    D = np.maximum(D, 0.0)
    return DescriptiveResult(
        name="simulate_2d_unfolding",
        value=D,
        extra={"X_resp": X_resp, "X_stim": X_stim, "n_resp": n_resp, "n_stim": n_stim},
    )


sim2d = simulate_2d_unfolding


def cheatsheet() -> str:
    return "simulate_2d_unfolding({}) -> Simulate 2D unfolding data. 'Ultra Instinct.' -- Goku, Drago"

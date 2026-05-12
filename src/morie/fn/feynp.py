# morie.fn -- function file (hadesllm/morie)
"""Feynman path integral (1D harmonic oscillator)."""

__all__ = ["feynp"]

import numpy as np


def feynp(
    x_i: float,
    x_f: float,
    T: float,
    m: float = 1.0,
    omega: float = 1.0,
    n_paths: int = 10000,
    n_slices: int = 50,
    seed: int = 42,
    hbar: float = 1.0,
) -> dict:
    r"""
    Evaluate the Feynman path integral for a 1D harmonic oscillator
    using Monte Carlo sampling, and compare with the exact propagator.

    The propagator:

    .. math::

        K(x_f, x_i; T) = \\int \\mathcal{D}[x(t)]
        \\exp\\left(\\frac{i}{\\hbar} S[x(t)]\\right)

    Exact result (harmonic oscillator):

    .. math::

        K = \\sqrt{\\frac{m\\omega}{2\\pi i \\hbar \\sin(\\omega T)}}
        \\exp\\left(\\frac{im\\omega}{2\\hbar\\sin(\\omega T)}
        \\left[(x_i^2 + x_f^2)\\cos(\\omega T) - 2 x_i x_f\\right]\\right)

    Parameters
    ----------
    x_i : float
        Initial position.
    x_f : float
        Final position.
    T : float
        Time duration (> 0).
    m : float
        Mass.
    omega : float
        Angular frequency.
    n_paths : int
        Number of Monte Carlo paths.
    n_slices : int
        Number of time slices.
    seed : int
        Random seed.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    dict
        Keys: propagator_mc (complex), propagator_exact (complex),
        classical_action, relative_error.
    """
    if T <= 0:
        raise ValueError("T must be > 0.")

    rng = np.random.default_rng(seed)
    dt = T / n_slices

    S_cl = (
        m * omega / (2.0 * np.sin(omega * T))
        * ((x_i ** 2 + x_f ** 2) * np.cos(omega * T) - 2.0 * x_i * x_f)
    )

    prefactor = np.sqrt(m * omega / (2.0 * np.pi * 1j * hbar * np.sin(omega * T + 1e-30)))
    K_exact = prefactor * np.exp(1j * S_cl / hbar)

    sigma = np.sqrt(hbar * dt / m)
    weights = np.zeros(n_paths, dtype=complex)

    for p in range(n_paths):
        path = np.zeros(n_slices + 1)
        path[0] = x_i
        path[-1] = x_f
        for k in range(1, n_slices):
            frac = k / n_slices
            path[k] = x_i * (1 - frac) + x_f * frac + rng.normal(0, sigma)

        S = 0.0
        for k in range(n_slices):
            v = (path[k + 1] - path[k]) / dt
            x_mid = 0.5 * (path[k] + path[k + 1])
            S += 0.5 * m * v ** 2 - 0.5 * m * omega ** 2 * x_mid ** 2
        S *= dt

        S_free = 0.0
        for k in range(n_slices):
            v = (path[k + 1] - path[k]) / dt
            S_free += 0.5 * m * v ** 2
        S_free *= dt

        weights[p] = np.exp(1j * (S - S_free) / hbar)

    norm = (m / (2.0 * np.pi * 1j * hbar * dt + 1e-300)) ** (n_slices / 2.0)
    K_mc = norm * np.mean(weights)

    rel_err = abs(abs(K_mc) - abs(K_exact)) / (abs(K_exact) + 1e-300)

    return {
        "propagator_mc": complex(K_mc),
        "propagator_exact": complex(K_exact),
        "classical_action": float(S_cl),
        "relative_error": float(rel_err),
    }

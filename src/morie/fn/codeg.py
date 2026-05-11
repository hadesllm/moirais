# morie.fn — function file (hadesllm/morie)
"""An unexamined life is not worth living. — Socrates"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def wigner_semicircle(
    n: int = 500,
    *,
    seed: int | None = 42,
    n_bins: int = 50,
) -> DescriptiveResult:
    """Generate a Wigner random matrix and verify the semicircle law.

    Creates an n x n symmetric random matrix from the Gaussian Orthogonal
    Ensemble (GOE) and computes the eigenvalue density, comparing it to the
    theoretical Wigner semicircle distribution.

    Parameters
    ----------
    n : int
        Matrix dimension.
    seed : int or None
        Random seed.
    n_bins : int
        Number of histogram bins for the empirical density.

    Returns
    -------
    DescriptiveResult
        ``value`` is the Kolmogorov-Smirnov statistic between empirical
        and theoretical CDF; ``extra`` has eigenvalue statistics.

    References
    ----------
    Wigner, E. P. (1958). On the distribution of the roots of certain
    symmetric matrices. Annals of Mathematics, 67(2), 325-327.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    rng = np.random.default_rng(seed)

    M = rng.standard_normal((n, n))
    M = (M + M.T) / (2 * np.sqrt(n))
    eigvals = np.linalg.eigvalsh(M)

    R = 2.0
    x_grid = np.linspace(-R, R, 200)
    semicircle_pdf = np.where(
        np.abs(x_grid) <= R,
        2 / (np.pi * R**2) * np.sqrt(np.maximum(R**2 - x_grid**2, 0)),
        0.0,
    )

    eigvals_sorted = np.sort(eigvals)
    ecdf = np.arange(1, n + 1) / n

    def _wigner_cdf(x):
        if x <= -R:
            return 0.0
        if x >= R:
            return 1.0
        return 0.5 + x * np.sqrt(R**2 - x**2) / (np.pi * R**2) + np.arcsin(x / R) / np.pi

    tcdf = np.array([_wigner_cdf(e) for e in eigvals_sorted])
    ks_stat = float(np.max(np.abs(ecdf - tcdf)))

    return DescriptiveResult(
        name="Wigner Semicircle",
        value=ks_stat,
        extra={
            "n": n,
            "eigenvalue_mean": float(eigvals.mean()),
            "eigenvalue_std": float(eigvals.std()),
            "eigenvalue_min": float(eigvals.min()),
            "eigenvalue_max": float(eigvals.max()),
            "theoretical_radius": R,
            "ks_statistic": ks_stat,
        },
    )


codeg = wigner_semicircle


def cheatsheet() -> str:
    return "wigner_semicircle({}) -> Random matrix theory (Wigner semicircle). 'It is the questio"

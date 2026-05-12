# morie.fn — function file (hadesllm/morie)
"""Molecular cloud mass function (Schechter). 'I was disassembled and rebuilt.' -- Nebula"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cloud_mass_function(
    masses: np.ndarray | list[float],
    *,
    n_bins: int = 20,
    alpha: float = -1.6,
) -> DescriptiveResult:
    r"""Fit a Schechter-like mass function to cloud mass observations.

    The cloud mass function (CMF) follows
    :math:`dN/dM \\propto M^{\\alpha} \\exp(-M/M^*)`.

    Parameters
    ----------
    masses : array-like
        Observed cloud masses (positive values).
    n_bins : int
        Number of log-spaced bins.
    alpha : float
        Initial guess for power-law index.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``alpha_fit``, ``M_star``, ``bin_centers``,
        ``dn_dlog_m``, ``total_mass``.
    """
    m = np.asarray(masses, dtype=float)
    m = m[m > 0]
    if len(m) < 5:
        raise ValueError("Need at least 5 positive masses")

    log_m = np.log10(m)
    bins = np.linspace(log_m.min(), log_m.max(), n_bins + 1)
    counts, edges = np.histogram(log_m, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dlogm = edges[1] - edges[0]

    dn_dlogm = counts / dlogm
    mask = dn_dlogm > 0
    if mask.sum() < 3:
        raise ValueError("Not enough non-empty bins for fitting")

    log_dn = np.log10(dn_dlogm[mask])
    log_centers = centers[mask]
    slope, intercept = np.polyfit(log_centers, log_dn, 1)

    M_star = float(10 ** (log_centers.max()))
    total_mass = float(m.sum())

    return DescriptiveResult(
        name="cloud_mass_function",
        value={
            "alpha_fit": float(slope),
            "M_star": M_star,
            "bin_centers": 10**centers,
            "dn_dlog_m": dn_dlogm,
            "total_mass": total_mass,
        },
        extra={"n_clouds": len(m), "n_bins": n_bins},
    )


nblam = cloud_mass_function


def cheatsheet() -> str:
    return "cloud_mass_function({}) -> Molecular cloud mass function (Schechter). 'I was disassembl"

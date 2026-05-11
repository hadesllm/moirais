"""Stepped-wedge trial design calculations."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def stepped_wedge_design(clusters: int, steps: int, cluster_size: int, effect_size: float, icc: float = 0.05, alpha: float = 0.05, cdf=None) -> ESRes:
    """Power and design parameters for a stepped-wedge cluster trial.

    In a stepped-wedge design, all clusters start as controls and
    sequentially cross over to intervention at each step.

    Parameters
    ----------
    clusters : int
        Total number of clusters.
    steps : int
        Number of crossover steps.
    cluster_size : int
        Individuals per cluster per period.
    effect_size : float
        Expected treatment effect (difference in proportions or means).
    icc : float, default 0.05
        Intra-cluster correlation.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    ESRes
        estimate is approximate power.

    References
    ----------
    Hussey, M. A. & Hughes, J. P. (2007). Design and analysis of
    stepped wedge cluster randomized trials. Contemporary Clinical
    Trials, 28(2), 182-191.
    """
    if clusters < 2 or steps < 1:
        raise ValueError("Need >= 2 clusters and >= 1 step")
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")

    K = clusters
    S = steps
    m = cluster_size
    T = S + 1

    deff = 1 + (m - 1) * icc
    sigma2 = 1.0
    tau2 = icc * sigma2
    sigma_e2 = sigma2 - tau2

    U = K * m * S * (S + 2) / (6 * (S + 1))
    W = K * m**2 * S**2 / (4 * T * (sigma_e2 / m + tau2))
    var_theta = 1.0 / max(W, 1e-10) if W > 0 else np.inf

    se_theta = np.sqrt(var_theta)
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    if se_theta > 0 and np.isfinite(se_theta):
        z_power = abs(effect_size) / se_theta - z_alpha
        power = float(stats.norm.cdf(z_power))
    else:
        power = 0.0

    return ESRes(
        measure="stepped_wedge",
        estimate=power,
        extra={
            "clusters": K,
            "steps": S,
            "periods": T,
            "cluster_size": m,
            "DEFF": float(deff),
            "effect_size": effect_size,
            "icc": icc,
            "se_theta": float(se_theta),
            "power": power,
        },
    )


stpwz = stepped_wedge_design


def cheatsheet() -> str:
    return "stepped_wedge_design({}) -> Stepped-wedge trial power calculation."

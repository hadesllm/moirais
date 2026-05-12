# morie.fn -- function file (hadesllm/morie)
"""Cluster randomized trial sample size calculation."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def cluster_trial_size(
    p1: float,
    p2: float,
    icc: float,
    cluster_size: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> ESRes:
    """Sample size for a cluster randomized trial.

    Accounts for the design effect (DEFF = 1 + (m-1)*ICC) which
    inflates the required sample size due to within-cluster correlation.

    Parameters
    ----------
    p1 : float
        Expected proportion in control arm.
    p2 : float
        Expected proportion in intervention arm.
    icc : float
        Intra-cluster correlation coefficient.
    cluster_size : int
        Number of individuals per cluster.
    alpha : float, default 0.05
        Significance level.
    power : float, default 0.80
        Desired power.

    Returns
    -------
    ESRes
        estimate is number of clusters per arm.

    References
    ----------
    Donner, A. & Klar, N. (2000). *Design and Analysis of Cluster
    Randomization Trials in Health Research*. Arnold, Ch. 3.
    """
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("p1 and p2 must be in (0, 1)")
    if icc < 0 or icc > 1:
        raise ValueError("ICC must be in [0, 1]")
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p_bar = (p1 + p2) / 2
    n_ind = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) /
             (p1 - p2)) ** 2

    deff = 1 + (cluster_size - 1) * icc
    n_adj = n_ind * deff
    n_clusters = int(np.ceil(n_adj / cluster_size))

    return ESRes(
        measure="cluster_trial_size",
        estimate=float(n_clusters),
        extra={
            "clusters_per_arm": n_clusters,
            "total_clusters": 2 * n_clusters,
            "individuals_per_arm": n_clusters * cluster_size,
            "DEFF": float(deff),
            "n_individual": float(n_ind),
            "icc": icc,
            "cluster_size": cluster_size,
        },
    )


cltrn = cluster_trial_size


def cheatsheet() -> str:
    return "cluster_trial_size({}) -> Cluster randomized trial sample size."

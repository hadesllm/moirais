# moirais.fn — function file (hadesllm/moirais)
"""Small-world coefficient."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes
from .netpl import network_path_length
from .ntccf import network_clustering_coeff


def network_small_world(
    adj: np.ndarray,
    *,
    n_random: int = 10,
    seed: int | None = None,
) -> ESRes:
    """Small-world coefficient sigma = (C/C_rand) / (L/L_rand).

    Parameters
    ----------
    adj : (n, n) binary adjacency
    n_random : int
        Number of random graphs for comparison.
    seed : int, optional

    Returns
    -------
    ESRes
    """
    A = (np.asarray(adj, dtype=float) > 0).astype(float)
    np.fill_diagonal(A, 0)
    n = A.shape[0]

    C_obs = network_clustering_coeff(A).estimate
    L_obs = network_path_length(A).estimate

    rng = np.random.default_rng(seed)
    edges = np.sum(A) / 2
    p_edge = edges / (n * (n - 1) / 2) if n > 1 else 0.0

    C_rands = []
    L_rands = []
    for _ in range(n_random):
        R = (rng.random((n, n)) < p_edge).astype(float)
        R = np.triu(R, 1)
        R = R + R.T
        C_rands.append(network_clustering_coeff(R).estimate)
        L_rands.append(network_path_length(R).estimate)

    C_rand = float(np.mean(C_rands)) if C_rands else 1e-12
    L_rand = float(np.mean(L_rands)) if L_rands else 1e-12

    gamma = C_obs / (C_rand + 1e-12)
    lam = L_obs / (L_rand + 1e-12)
    sigma = gamma / (lam + 1e-12)

    return ESRes(
        measure="small_world_sigma",
        estimate=float(sigma),
        n=n,
        extra={
            "C_obs": C_obs,
            "L_obs": L_obs,
            "C_rand": C_rand,
            "L_rand": L_rand,
            "gamma": float(gamma),
            "lambda": float(lam),
        },
    )


netsw = network_small_world


def cheatsheet() -> str:
    return "network_small_world({}) -> Small-world coefficient."

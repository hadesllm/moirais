# morie.fn -- function file (hadesllm/morie)
"""Exchangeable bootstrap weights."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["exchw"]


def exchw(
    n: int,
    *,
    n_boot: int = 1000,
    distribution: str = "dirichlet",
    seed: int | None = None,
) -> dict[str, Any]:
    r"""
    Generate exchangeable bootstrap weight matrices.

    Exchangeable bootstrap weights :math:`(w_1, \ldots, w_n)` satisfy
    :math:`\sum w_i = n` and are exchangeable (identically distributed,
    any permutation equally likely).

    :param n: Sample size.
    :param n_boot: Number of bootstrap replications. Default 1000.
    :param distribution: Weight distribution: ``"dirichlet"`` (default),
        ``"exponential"``, ``"multinomial"``.
    :param seed: Random seed.
    :return: Dict with ``weights`` (n_boot x n matrix), ``n``, ``n_boot``,
        ``distribution``.
    :raises ValueError: If n < 1 or n_boot < 1.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 19. Springer.
    Praestgaard & Wellner (1993). Exchangeably weighted bootstraps.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}.")

    rng = np.random.default_rng(seed)
    weights = np.zeros((n_boot, n))

    if distribution == "dirichlet":
        for b in range(n_boot):
            w = rng.dirichlet(np.ones(n))
            weights[b] = w * n
    elif distribution == "exponential":
        for b in range(n_boot):
            w = rng.exponential(1.0, size=n)
            weights[b] = w / np.mean(w)
    elif distribution == "multinomial":
        for b in range(n_boot):
            counts = rng.multinomial(n, np.ones(n) / n)
            weights[b] = counts.astype(float)
    else:
        raise ValueError(f"distribution must be 'dirichlet', 'exponential', or 'multinomial', got '{distribution}'.")

    return {
        "weights": weights,
        "n": n,
        "n_boot": n_boot,
        "distribution": distribution,
    }


def cheatsheet() -> str:
    return "exchw({n}) -> Exchangeable bootstrap weights."

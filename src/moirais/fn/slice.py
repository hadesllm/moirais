"""Slice sampling."""

from __future__ import annotations

__all__ = ["slice_sampler", "slice"]

from collections.abc import Callable
from typing import Any

import numpy as np


def slice_sampler(
    log_target: Callable[[float], float],
    init: float = 0.0,
    *,
    width: float = 1.0,
    n_iter: int = 5000,
    max_steps: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Univariate slice sampler with stepping-out and shrinking.

    Draws samples from a univariate distribution specified by its
    (unnormalized) log-density using Neal's stepping-out procedure
    to find the slice interval, then shrinks until a valid point
    is found.

    Parameters
    ----------
    log_target : callable
        Univariate log-density function (unnormalized OK).
    init : float
        Initial value.
    width : float
        Initial bracket width for stepping out.
    n_iter : int
        Number of samples to draw.
    max_steps : int
        Maximum stepping-out expansions per side.
    seed : int
        Random seed.

    Returns
    -------
    dict
        samples : ndarray (n_iter,)
        n_iter : int

    References
    ----------
    Neal, R. M. (2003). Slice sampling. *Annals of Statistics*,
    31(3), 705--767.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")
    if width <= 0:
        raise ValueError("width must be > 0.")

    rng = np.random.default_rng(seed)
    x = float(init)
    samples = np.empty(n_iter)

    for i in range(n_iter):
        log_y = log_target(x) + np.log(rng.uniform())

        L = x - width * rng.uniform()
        R = L + width

        j = int(np.floor(max_steps * rng.uniform()))
        k = max_steps - 1 - j

        while j > 0 and log_target(L) > log_y:
            L -= width
            j -= 1
        while k > 0 and log_target(R) > log_y:
            R += width
            k -= 1

        while True:
            x_prop = L + rng.uniform() * (R - L)
            if log_target(x_prop) >= log_y:
                x = x_prop
                break
            if x_prop < x:
                L = x_prop
            else:
                R = x_prop

        samples[i] = x

    return {
        "samples": samples,
        "n_iter": n_iter,
    }


slice = slice_sampler


def cheatsheet() -> str:
    return "slice_sampler(log_target, init) -> Univariate slice sampler."

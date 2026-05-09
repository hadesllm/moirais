"""Slice sampler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def slice_sampler(
    log_target: Callable[[float], float],
    init: float = 0.0,
    *,
    width: float = 1.0,
    n_iter: int = 10000,
    max_steps: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Univariate slice sampler with stepping-out and shrinking.

    :param log_target: Log-density of target (univariate, unnormalized OK).
    :param init: Initial value.
    :param width: Initial bracket width.
    :param n_iter: Number of iterations.
    :param max_steps: Maximum stepping-out steps.
    :param seed: Random seed.
    :return: Dictionary with samples array.

    References
    ----------
    Neal, R. M. (2003). *Annals of Statistics*, 31(3), 705--741.
    """
    rng = np.random.default_rng(seed)
    x = float(init)
    samples = np.empty(n_iter)

    for i in range(n_iter):
        log_y = log_target(x) + np.log(rng.uniform())

        L = x - width * rng.uniform()
        R = L + width

        steps = 0
        while log_target(L) > log_y and steps < max_steps:
            L -= width
            steps += 1
        steps = 0
        while log_target(R) > log_y and steps < max_steps:
            R += width
            steps += 1

        while True:
            x_new = L + rng.uniform() * (R - L)
            if log_target(x_new) > log_y:
                x = x_new
                break
            if x_new < x:
                L = x_new
            else:
                R = x_new

        samples[i] = x

    return {
        "samples": samples,
        "n_iter": n_iter,
    }


slisp = slice_sampler


def cheatsheet() -> str:
    return "slice_sampler({}) -> Slice sampler."

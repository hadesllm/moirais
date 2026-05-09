# moirais.fn — function file (hadesllm/moirais)
"""Latin Hypercube Sampling."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def latin_hypercube(
    n_samples: int,
    n_dims: int,
    seed: int = 42,
) -> DescriptiveResult:
    """Generate a Latin Hypercube Sample on [0, 1]^d.

    Divides each dimension into *n* equal strata and places exactly
    one sample per stratum, with random placement within each cell.

    Parameters
    ----------
    n_samples : int
        Number of samples (n >= 1).
    n_dims : int
        Number of dimensions (d >= 1).
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    DescriptiveResult
        ``value`` is the (n_samples x n_dims) sample matrix.
        ``extra`` has ``n_samples``, ``n_dims``, ``seed``.

    References
    ----------
    McKay, M. D., Beckman, R. J., & Conover, W. J. (1979).
    A comparison of three methods for selecting values of input
    variables in the analysis of output from a computer code.
    *Technometrics*, 21(2), 239--245.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    if n_dims < 1:
        raise ValueError(f"n_dims must be >= 1, got {n_dims}.")

    rng = np.random.default_rng(seed)
    result = np.zeros((n_samples, n_dims))

    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            result[i, d] = (perm[i] + rng.uniform()) / n_samples

    return DescriptiveResult(
        name="LatinHypercube",
        value=result,
        extra={"n_samples": n_samples, "n_dims": n_dims, "seed": seed},
    )


latns = latin_hypercube


def cheatsheet() -> str:
    return "latin_hypercube({}) -> Latin Hypercube Sampling."

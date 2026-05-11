# morie.fn — function file (hadesllm/morie)
"""Initial start values for MDS. 'WRYYYYY!' -- DIO, JoJo's Bizarre Adventure"""

from __future__ import annotations

from ._containers import DescriptiveResult


def initial_start_values(n, n_dims=2, method="random", seed=42):
    """Generate starting configuration for MDS.

    Parameters
    ----------
    n : int
        Number of objects.
    n_dims : int
        Embedding dimensionality.
    method : str
        'random' or 'grid'.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        value = initial coordinate matrix (n x n_dims).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    if method == "grid":
        side = int(np.ceil(n ** (1.0 / n_dims)))
        axes = [np.linspace(-1, 1, side) for _ in range(n_dims)]
        grid = np.array(np.meshgrid(*axes)).reshape(n_dims, -1).T
        X = grid[:n]
    else:
        X = rng.standard_normal((n, n_dims))
    return DescriptiveResult(name="initial_start_values", value=X, extra={"method": method, "n": n, "n_dims": n_dims})


inits = initial_start_values


def cheatsheet() -> str:
    return "initial_start_values({}) -> Initial start values for MDS. 'WRYYYYY!' -- DIO, JoJo's Biza"

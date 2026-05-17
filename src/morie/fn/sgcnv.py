"""Convolution representation of a random field."""

from __future__ import annotations

from ._containers import DescriptiveResult


def convolution_representation(kernel, white_noise, coords):
    """Generate a spatial field via convolution of a kernel with white noise.

    .. epigraph:: In the midst of chaos, there is also opportunity. -- Sun Tzu

    Parameters
    ----------
    kernel : callable
        Kernel function k(distance) -> weight.
    white_noise : array_like
        White noise values at source locations, shape ``(m,)``.
    coords : array_like
        Target coordinates, shape ``(n, 2)``.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    wn = np.asarray(white_noise, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    m = len(wn)

    rng = np.random.default_rng(42)
    source_coords = rng.uniform(coords.min(axis=0), coords.max(axis=0), size=(m, 2))

    field = np.zeros(n)
    for i in range(n):
        dists = np.sqrt(np.sum((source_coords - coords[i]) ** 2, axis=1))
        weights = np.array([kernel(d) for d in dists])
        field[i] = np.sum(weights * wn)

    return DescriptiveResult(
        name="convolution_representation",
        value=float(np.var(field)),
        extra={
            "field": field,
            "mean": float(np.mean(field)),
            "variance": float(np.var(field)),
            "n_sources": m,
        },
    )


sgcnv = convolution_representation


def cheatsheet() -> str:
    return "convolution_representation({}) -> Convolution representation of a random field."

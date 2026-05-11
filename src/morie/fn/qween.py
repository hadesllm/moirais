# morie.fn — function file (hadesllm/morie)
"""Queen contiguity spatial weights matrix from adjacency list."""

import numpy as np

from ._containers import DescriptiveResult


def queen_weights(
    adjacency: list[list[int]],
) -> DescriptiveResult:
    """
    Construct a Queen contiguity spatial weights matrix from an adjacency list.

    Queen contiguity defines two spatial units as neighbors if they share any
    boundary point (edge or vertex). This function takes a pre-computed
    adjacency list and returns the binary weight matrix.

    :param adjacency: List of length *n* where ``adjacency[i]`` is a list of
        integer indices of the neighbors of unit *i*.
    :return: :class:`DescriptiveResult` with value = binary weight matrix (n, n).
    :raises ValueError: If adjacency contains out-of-range indices.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.

    LeSage, J. & Pace, R. K. (2009). *Introduction to Spatial Econometrics*.
    CRC Press.
    """
    n = len(adjacency)
    W = np.zeros((n, n), dtype=float)

    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if j < 0 or j >= n:
                raise ValueError(f"Neighbor index {j} out of range [0, {n}).")
            if j != i:
                W[i, j] = 1.0

    return DescriptiveResult(
        name="Queen Contiguity Weights",
        value=W,
        extra={"n": n},
    )


qween_fn = queen_weights


def cheatsheet() -> str:
    return "queen_weights({}) -> Queen contiguity spatial weights matrix from adjacency list."

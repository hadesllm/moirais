# morie.fn — function file (hadesllm/morie)
"""Plackett-Burman screening design."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plackett_burman(n_factors: int) -> DescriptiveResult:
    """Generate a Plackett-Burman screening design.

    Plackett-Burman designs are two-level fractional factorials with
    *N* runs (a multiple of 4) that can screen up to *N - 1* factors.
    The design is constructed from a cyclic generator row.

    Parameters
    ----------
    n_factors : int
        Number of factors to screen (1 <= n_factors <= 23).
        The number of runs *N* is the smallest multiple of 4 that
        exceeds n_factors.

    Returns
    -------
    DescriptiveResult
        ``value`` is the design matrix (N x n_factors), coded -1/+1.
        ``extra`` has ``n_runs`` and ``n_factors``.

    Raises
    ------
    ValueError
        If n_factors < 1 or > 23.

    References
    ----------
    Plackett, R. L., & Burman, J. P. (1946). The design of optimum
    multifactorial experiments. *Biometrika*, 33(4), 305--325.
    """
    if n_factors < 1 or n_factors > 23:
        raise ValueError(f"n_factors must be in [1, 23], got {n_factors}.")

    generators = {
        4: [1, 1, -1],
        8: [1, 1, 1, -1, 1, -1, -1],
        12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
        16: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],
        20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],
        24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1],
    }

    N = 4
    while n_factors > N - 1:
        N += 4
    if N > 24:
        raise ValueError(f"n_factors={n_factors} requires N={N} > 24, unsupported.")

    gen = generators[N]
    k = len(gen)
    rows = []
    for i in range(k):
        row = gen[i:] + gen[:i]
        rows.append(row)
    rows.append([-1] * k)

    design = np.array(rows, dtype=np.float64)[:, :n_factors]

    return DescriptiveResult(
        name="PlackettBurman",
        value=design,
        extra={"n_runs": design.shape[0], "n_factors": n_factors},
    )


plakb = plackett_burman


def cheatsheet() -> str:
    return "plackett_burman({}) -> Plackett-Burman screening design."

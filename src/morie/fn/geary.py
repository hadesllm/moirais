# morie.fn -- function file (hadesllm/morie)
"""Geary's C spatial autocorrelation statistic."""

import numpy as np

from ._containers import SpatialResult


def gearys_c(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    nperm: int = 999,
    seed: int | None = None,
) -> SpatialResult:
    r"""
    Compute Geary's C ratio for spatial autocorrelation.

    .. math::

        C = \\frac{(n-1)}{2W} \\cdot
            \\frac{\\sum_i \\sum_j w_{ij}(x_i - x_j)^2}
                  {\\sum_i (x_i - \\bar{x})^2}

    Values of *C* < 1 indicate positive spatial autocorrelation (clustering),
    *C* > 1 indicates negative autocorrelation (dispersion), and *C* = 1
    indicates spatial randomness.

    :param values: Observed values array (n,).
    :param weights: Spatial weights matrix (n, n).
    :param nperm: Number of permutations for pseudo p-value.
    :param seed: Random seed for reproducibility.
    :return: :class:`SpatialResult` with statistic and p_value.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Geary, R. C. (1954). The contiguity ratio and statistical mapping.
    *The Incorporated Statistician*, 5(3), 115-145.
    https://doi.org/10.2307/2986645
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(values)
    if weights.shape != (n, n):
        raise ValueError(f"weights must be ({n}, {n}), got {weights.shape}.")

    xbar = values.mean()
    z = values - xbar
    W = weights.sum()
    ss = float(z @ z)

    if ss == 0 or W == 0:
        return SpatialResult(name="Geary's C", statistic=1.0, p_value=1.0, expected=1.0)

    diff_sq = (values[:, None] - values[None, :]) ** 2
    numer = float(np.sum(weights * diff_sq))
    C_obs = ((n - 1) / (2 * W)) * numer / ss

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(nperm):
        perm = rng.permutation(values)
        z_p = perm - perm.mean()
        ss_p = float(z_p @ z_p)
        if ss_p == 0:
            continue
        diff_p = (perm[:, None] - perm[None, :]) ** 2
        C_perm = ((n - 1) / (2 * W)) * float(np.sum(weights * diff_p)) / ss_p
        if C_perm <= C_obs:
            count += 1
    p_value = (count + 1) / (nperm + 1)

    return SpatialResult(
        name="Geary's C",
        statistic=C_obs,
        p_value=p_value,
        expected=1.0,
        extra={"n": n, "W": W, "nperm": nperm},
    )


geary_fn = gearys_c


def cheatsheet() -> str:
    return "gearys_c({}) -> Geary's C spatial autocorrelation statistic."

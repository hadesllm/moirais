# moirais.fn — function file (hadesllm/moirais)
"""Moran's I global spatial autocorrelation statistic."""

import numpy as np

from ._containers import SpatialResult


def morans_i(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    nperm: int = 999,
    seed: int | None = None,
) -> SpatialResult:
    """
    Compute Moran's I for global spatial autocorrelation.

    .. math::

        I = \\frac{n}{W} \\cdot
            \\frac{\\sum_i \\sum_j w_{ij}(x_i - \\bar{x})(x_j - \\bar{x})}
                  {\\sum_i (x_i - \\bar{x})^2}

    where :math:`W = \\sum_i \\sum_j w_{ij}`.

    Statistical significance is assessed via a permutation test: observed *I*
    is compared against the distribution of *I* under random spatial
    rearrangements of the values.

    :param values: Array of observed values (n,).
    :param weights: Spatial weights matrix (n, n). Need not be row-standardised.
    :param nperm: Number of random permutations for the pseudo p-value.
    :param seed: Random seed for reproducibility.
    :return: :class:`SpatialResult` with statistic, p_value, expected, variance.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Moran, P. A. P. (1950). Notes on continuous stochastic phenomena.
    *Biometrika*, 37(1/2), 17-23. https://doi.org/10.2307/2332142

    Cliff, A. D. & Ord, J. K. (1981). *Spatial Processes: Models &
    Applications*. Pion.
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
        return SpatialResult(name="Moran's I", statistic=0.0, p_value=1.0, expected=-1.0 / (n - 1), variance=0.0)

    numerator = float(z @ weights @ z)
    I_obs = (n / W) * numerator / ss

    E_I = -1.0 / (n - 1)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(nperm):
        perm = rng.permutation(z)
        I_perm = (n / W) * float(perm @ weights @ perm) / float(perm @ perm)
        if I_perm >= I_obs:
            count += 1
    p_value = (count + 1) / (nperm + 1)

    return SpatialResult(
        name="Moran's I",
        statistic=I_obs,
        p_value=p_value,
        expected=E_I,
        variance=None,
        extra={"n": n, "W": W, "nperm": nperm},
    )


moran_fn = morans_i


def cheatsheet() -> str:
    return "morans_i({}) -> Moran's I global spatial autocorrelation statistic."

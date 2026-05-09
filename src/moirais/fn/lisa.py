# moirais.fn — function file (hadesllm/moirais)
"""Local Moran's I (LISA) statistic."""

import numpy as np

from ._containers import SpatialResult


def local_morans_i(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    nperm: int = 999,
    seed: int | None = None,
) -> SpatialResult:
    """
    Compute Local Indicators of Spatial Association (LISA).

    For each observation *i*:

    .. math::

        I_i = \\frac{z_i}{m_2} \\sum_j w_{ij} z_j

    where :math:`z_i = x_i - \\bar{x}` and :math:`m_2 = \\frac{1}{n}\\sum z_i^2`.

    A global Moran's I is also returned as the mean of the local values
    (weighted by the weights matrix structure).

    :param values: Observed values array (n,).
    :param weights: Spatial weights matrix (n, n). Typically row-standardised.
    :param nperm: Number of permutations for pseudo p-value of global I.
    :param seed: Random seed for reproducibility.
    :return: :class:`SpatialResult` with local_values array of local I_i,
        and statistic = global Moran's I.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Anselin, L. (1995). Local indicators of spatial association -- LISA.
    *Geographical Analysis*, 27(2), 93-115.
    https://doi.org/10.1111/j.1538-4632.1995.tb00338.x
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(values)
    if weights.shape != (n, n):
        raise ValueError(f"weights must be ({n}, {n}), got {weights.shape}.")

    xbar = values.mean()
    z = values - xbar
    m2 = float(np.mean(z**2))

    if m2 == 0:
        local_vals = np.zeros(n)
        return SpatialResult(
            name="Local Moran's I (LISA)", statistic=0.0, p_value=1.0, expected=0.0, local_values=local_vals
        )

    wz = weights @ z
    local_I = (z / m2) * wz

    W = weights.sum()
    ss = float(z @ z)
    I_global = (n / W) * float(z @ weights @ z) / ss if W != 0 and ss != 0 else 0.0

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(nperm):
        perm = rng.permutation(z)
        ss_p = float(perm @ perm)
        if ss_p == 0 or W == 0:
            continue
        I_perm = (n / W) * float(perm @ weights @ perm) / ss_p
        if I_perm >= I_global:
            count += 1
    p_value = (count + 1) / (nperm + 1)

    return SpatialResult(
        name="Local Moran's I (LISA)",
        statistic=I_global,
        p_value=p_value,
        expected=-1.0 / (n - 1),
        local_values=local_I,
        extra={"n": n},
    )


lisa_fn = local_morans_i


def cheatsheet() -> str:
    return "local_morans_i({}) -> Local Moran's I (LISA) statistic."

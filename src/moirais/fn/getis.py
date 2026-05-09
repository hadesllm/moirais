# moirais.fn — function file (hadesllm/moirais)
"""Getis-Ord Gi* hot spot statistic."""

import numpy as np

from ._containers import SpatialResult


def getis_ord_gi_star(
    values: np.ndarray,
    weights: np.ndarray,
) -> SpatialResult:
    """
    Compute the Getis-Ord :math:`G_i^*` statistic for hot/cold spot detection.

    For each observation *i*:

    .. math::

        G_i^* = \\frac{\\sum_j w_{ij} x_j - \\bar{x} \\sum_j w_{ij}}
                      {s \\sqrt{\\frac{n \\sum_j w_{ij}^2
                      - \\left(\\sum_j w_{ij}\\right)^2}{n - 1}}}

    where the sums include *j = i* (distinguishing :math:`G_i^*` from
    :math:`G_i`). Under the null hypothesis of no spatial clustering, each
    :math:`G_i^*` follows a standard normal distribution.

    :param values: Observed values array (n,).
    :param weights: Spatial weights matrix (n, n). The diagonal may be nonzero
        (self-weights are included in :math:`G_i^*`).
    :return: :class:`SpatialResult` with local_values = array of z-scores,
        statistic = max absolute z-score.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Getis, A. & Ord, J. K. (1992). The analysis of spatial association by use
    of distance statistics. *Geographical Analysis*, 24(3), 189-206.
    https://doi.org/10.1111/j.1538-4632.1992.tb00261.x

    Ord, J. K. & Getis, A. (1995). Local spatial autocorrelation statistics:
    distributional issues and an application. *Geographical Analysis*, 27(4),
    286-306. https://doi.org/10.1111/j.1538-4632.1995.tb00912.x
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(values)
    if weights.shape != (n, n):
        raise ValueError(f"weights must be ({n}, {n}), got {weights.shape}.")

    xbar = values.mean()
    s = values.std(ddof=0)

    if s == 0:
        return SpatialResult(name="Getis-Ord Gi*", statistic=0.0, p_value=1.0, local_values=np.zeros(n))

    gi_star = np.zeros(n)
    for i in range(n):
        wi = weights[i]
        sum_wij = wi.sum()
        sum_wij2 = float(wi @ wi)
        numer = float(wi @ values) - xbar * sum_wij
        denom = s * np.sqrt((n * sum_wij2 - sum_wij**2) / (n - 1))
        gi_star[i] = numer / denom if denom > 0 else 0.0

    return SpatialResult(
        name="Getis-Ord Gi*",
        statistic=float(np.max(np.abs(gi_star))),
        p_value=None,
        local_values=gi_star,
        extra={"n": n, "mean": xbar, "sd": s},
    )


getis_fn = getis_ord_gi_star


def cheatsheet() -> str:
    return "getis_ord_gi_star({}) -> Getis-Ord Gi* hot spot statistic."

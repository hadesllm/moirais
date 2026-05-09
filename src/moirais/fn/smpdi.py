"""Simpson's diversity index."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There is always a bigger fish. -- Qui-Gon Jinn"


def simpson_diversity(abundances, **kwargs) -> DescriptiveResult:
    """
    Compute Simpson's diversity index (1 - D).

    Simpson's concentration is:

    .. math::

        D = \\sum_{i=1}^{S} p_i^2

    The diversity index is :math:`1 - D` (probability that two randomly
    chosen individuals belong to different species). The reciprocal
    :math:`1/D` is also reported.

    :param abundances: Array-like of species counts (non-negative).
    :return: DescriptiveResult with 1-D as value.
    :raises ValueError: If any abundance is negative or all are zero.

    References
    ----------
    Simpson, E. H. (1949). Measurement of diversity. *Nature*, 163, 688.
    """
    abundances = np.asarray(abundances, dtype=np.float64)
    if np.any(abundances < 0):
        raise ValueError("Abundances must be non-negative.")
    total = float(np.sum(abundances))
    if total == 0:
        raise ValueError("Total abundance must be > 0.")

    p = abundances[abundances > 0] / total
    D = float(np.sum(p**2))
    diversity = 1.0 - D
    reciprocal = 1.0 / D if D > 0 else float("inf")

    return DescriptiveResult(
        name="simpson_diversity",
        value=diversity,
        extra={
            "simpson_D": D,
            "diversity_1_minus_D": diversity,
            "reciprocal_1_over_D": reciprocal,
            "species_richness": int(np.sum(abundances > 0)),
            "total_abundance": total,
        },
    )


smpdi = simpson_diversity


def cheatsheet() -> str:
    return "simpson_diversity({}) -> Simpson's diversity index."

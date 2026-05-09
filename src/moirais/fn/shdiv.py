"""Shannon diversity index."""

import numpy as np

from ._containers import DescriptiveResult
def shannon_diversity(abundances, **kwargs) -> DescriptiveResult:
    """
    Compute the Shannon diversity index H'.

    .. math::

        H' = -\\sum_{i=1}^{S} p_i \\ln p_i

    where :math:`p_i = n_i / N` is the proportion of species *i* and
    *S* is species richness. Also returns evenness :math:`J = H' / \\ln S`.

    :param abundances: Array-like of species counts (non-negative integers).
    :return: DescriptiveResult with H' as value and evenness in extra.
    :raises ValueError: If any abundance is negative or all are zero.

    References
    ----------
    Shannon, C. E. (1948). A mathematical theory of communication.
    *Bell System Technical Journal*, 27(3), 379-423.
    Magurran, A. E. (2004). *Measuring Biological Diversity*. Blackwell.
    """
    abundances = np.asarray(abundances, dtype=np.float64)
    if np.any(abundances < 0):
        raise ValueError("Abundances must be non-negative.")
    total = float(np.sum(abundances))
    if total == 0:
        raise ValueError("Total abundance must be > 0.")

    p = abundances[abundances > 0] / total
    h_prime = float(-np.sum(p * np.log(p)))

    s = int(np.sum(abundances > 0))
    evenness = h_prime / np.log(s) if s > 1 else 1.0

    return DescriptiveResult(
        name="shannon_diversity",
        value=h_prime,
        extra={
            "H_prime": h_prime,
            "species_richness": s,
            "evenness_J": float(evenness),
            "total_abundance": total,
        },
    )


shdiv = shannon_diversity


def cheatsheet() -> str:
    return "shannon_diversity({}) -> Shannon diversity index."

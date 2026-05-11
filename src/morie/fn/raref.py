# morie.fn — function file (hadesllm/morie)
"""Rarefaction curve computation."""

import numpy as np
from scipy.special import comb

from ._containers import DescriptiveResult
def rarefaction(abundances, n_subsample=None, n_points: int = 20, **kwargs) -> DescriptiveResult:
    """
    Compute an individual-based rarefaction curve.

    Expected species richness at subsample size *n*:

    .. math::

        E[S_n] = S - \\sum_{i=1}^{S} \\frac{\\binom{N - N_i}{n}}{\\binom{N}{n}}

    where *N* is total individuals and :math:`N_i` is count of species *i*.

    :param abundances: Array-like of species counts (non-negative integers).
    :param n_subsample: Specific subsample size. If None, computes full curve.
    :param n_points: Number of points on the curve if n_subsample is None.
    :return: DescriptiveResult with expected richness at n_subsample.
    :raises ValueError: If n_subsample > total abundance.

    References
    ----------
    Hurlbert, S. H. (1971). The nonconcept of species diversity: a critique
    and alternative parameters. *Ecology*, 52(4), 577-586.
    Gotelli, N. J. & Colwell, R. K. (2001). Quantifying biodiversity.
    *Ecology Letters*, 4(4), 379-391.
    """
    abundances = np.asarray(abundances, dtype=np.float64)
    counts = abundances[abundances > 0]
    N = int(np.sum(counts))
    S = int(len(counts))

    if N == 0:
        raise ValueError("Total abundance must be > 0.")

    def _expected_richness(n):
        if n >= N:
            return float(S)
        result = S - np.sum([comb(N - ni, n, exact=True) / comb(N, n, exact=True) for ni in counts.astype(int)])
        return float(result)

    if n_subsample is not None:
        n_subsample = int(n_subsample)
        if n_subsample > N:
            raise ValueError(f"n_subsample ({n_subsample}) > total ({N}).")
        e_s = _expected_richness(n_subsample)
        sizes = np.array([n_subsample])
        curve = np.array([e_s])
    else:
        sizes = np.linspace(1, N, min(n_points, N)).astype(int)
        sizes = np.unique(sizes)
        curve = np.array([_expected_richness(n) for n in sizes])
        e_s = float(curve[-1])

    return DescriptiveResult(
        name="rarefaction",
        value=e_s,
        extra={
            "expected_richness": e_s,
            "subsample_sizes": sizes,
            "rarefaction_curve": curve,
            "total_N": N,
            "observed_S": S,
        },
    )


raref = rarefaction


def cheatsheet() -> str:
    return "rarefaction({}) -> Rarefaction curve computation."

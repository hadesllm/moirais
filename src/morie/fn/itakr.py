# morie.fn — function file (hadesllm/morie)
"""Itakura-Saito distance."""

import numpy as np

from ._containers import ESRes


def itakura_saito(p_spec, q_spec, **kwargs) -> ESRes:
    """
    Compute the Itakura-Saito distance between two power spectra.

    .. math::

        d_{IS}(P, Q) = \\frac{1}{K}\\sum_k
        \\left(\\frac{P(k)}{Q(k)} - \\ln\\frac{P(k)}{Q(k)} - 1\\right)

    A scale-invariant distance. Equals 0 iff P = c*Q for some c.

    :param p_spec: array-like, reference power spectrum.
    :param q_spec: array-like, comparison power spectrum.
    :return: ESRes with Itakura-Saito distance (>= 0).

    References
    ----------
    Itakura F (1968). Analysis synthesis telephony based on the
    maximum likelihood method. Reports of the 6th International
    Congress on Acoustics, C-5-5.
    """
    p = np.asarray(p_spec, dtype=np.float64).ravel()
    q = np.asarray(q_spec, dtype=np.float64).ravel()
    if len(p) != len(q):
        raise ValueError("p_spec and q_spec must have same length.")
    if np.any(p <= 0) or np.any(q <= 0):
        raise ValueError("Both spectra must be strictly positive.")
    ratio = p / q
    d = float(np.mean(ratio - np.log(ratio) - 1.0))
    return ESRes(
        measure="itakura_saito",
        estimate=d,
        n=len(p),
        extra={"n_bins": len(p)},
    )


itakr = itakura_saito


def cheatsheet() -> str:
    return "itakura_saito(p, q) -> Itakura-Saito spectral distance."

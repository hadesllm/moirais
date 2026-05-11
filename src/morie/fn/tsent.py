"""Tsallis entropy."""

import numpy as np

from ._containers import ESRes


def tsallis_entropy(x, q: float = 2.0, bins: int = 50, **kwargs) -> ESRes:
    """
    Compute Tsallis entropy of order q.

    .. math::

        S_q(X) = \\frac{1}{q-1}\\left(1 - \\sum_i p_i^q\\right)

    For q -> 1, recovers Shannon entropy.

    :param x: array-like data.
    :param q: Entropic index (q != 1). Default 2.0.
    :param bins: Histogram bins for continuous data.
    :return: ESRes with Tsallis entropy.

    References
    ----------
    Tsallis C (1988). Possible generalization of Boltzmann-Gibbs statistics.
    Journal of Statistical Physics, 52(1-2), 479-487.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 1:
        raise ValueError("Need at least 1 observation.")
    if abs(q - 1.0) < 1e-12:
        raise ValueError("q must not equal 1 (use Shannon entropy instead).")
    counts, _ = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    s = float((1.0 - np.sum(p ** q)) / (q - 1.0))
    return ESRes(
        measure="tsallis_entropy",
        estimate=s,
        n=len(x),
        extra={"q": q, "n_bins": bins},
    )


tsent = tsallis_entropy


def cheatsheet() -> str:
    return "tsallis_entropy(x, q=2) -> Tsallis entropy of order q."

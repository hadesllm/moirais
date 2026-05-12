# morie.fn -- function file (hadesllm/morie)
"""Renyi entropy."""

import numpy as np

from ._containers import ESRes
def renyi_entropy(x, alpha: float = 2.0, bins: int = 50, **kwargs) -> ESRes:
    r"""
    Compute Renyi entropy of order α.

    .. math::

        H_{\\alpha}(X) = \\frac{1}{1 - \\alpha} \\log_2 \\left(
        \\sum_i p_i^{\\alpha} \\right)

    For α -> 1 this converges to Shannon entropy. α = 2 gives
    collision entropy. α = 0 gives Hartley entropy.

    :param x: array-like of observations.
    :param alpha: Order of the Renyi entropy. Must be >= 0 and != 1.
    :param bins: Number of histogram bins.
    :return: ESRes with Renyi entropy in bits.

    References
    ----------
    Renyi A (1961). On measures of entropy and information.
    Proceedings of the Fourth Berkeley Symposium, 1, 547-561.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 1:
        raise ValueError("Need at least 1 observation.")
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")
    counts, _ = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    if abs(alpha - 1.0) < 1e-12:
        h = -float(np.sum(p * np.log2(p)))
    elif alpha == 0:
        h = float(np.log2(len(p)))
    else:
        h = float(np.log2(np.sum(p**alpha)) / (1 - alpha))
    return ESRes(
        measure="renyi_entropy",
        estimate=h,
        n=len(x),
        extra={"alpha": alpha, "bits": h, "n_bins": bins},
    )


renyi = renyi_entropy


def cheatsheet() -> str:
    return "renyi_entropy({}) -> Renyi entropy."

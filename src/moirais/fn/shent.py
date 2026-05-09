"""Shannon entropy."""

import numpy as np

from ._containers import ESRes
def shannon_entropy(x, bins: int = 50, **kwargs) -> ESRes:
    """
    Compute Shannon entropy of a discrete or discretised signal.

    .. math::

        H(X) = -\\sum_{i} p(x_i) \\log_2 p(x_i)

    :param x: array-like. If continuous, histogram-binned into *bins* bins.
    :param bins: Number of histogram bins for continuous data.
    :return: ESRes with entropy in bits.

    References
    ----------
    Shannon CE (1948). A mathematical theory of communication.
    Bell System Technical Journal, 27(3), 379-423.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 1:
        raise ValueError("Need at least 1 observation.")
    counts, _ = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    h = -float(np.sum(p * np.log2(p)))
    return ESRes(
        measure="shannon_entropy",
        estimate=h,
        n=len(x),
        extra={"bits": h, "nats": float(h * np.log(2)), "n_bins": bins},
    )


shent = shannon_entropy


def cheatsheet() -> str:
    return "shannon_entropy({}) -> Shannon entropy."

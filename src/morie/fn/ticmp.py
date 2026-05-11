"""Total information content."""

import numpy as np

from ._containers import ESRes


def total_information_content(x, bins: int = 50, **kwargs) -> ESRes:
    """
    Compute total information content (TIC) = N * H(X).

    Total information in a sample is the product of sample size and
    Shannon entropy, measured in bits.

    :param x: array-like data.
    :param bins: Number of histogram bins.
    :return: ESRes with total information content.

    References
    ----------
    Shannon CE (1948). A mathematical theory of communication.
    Bell System Technical Journal, 27(3), 379-423.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 1:
        raise ValueError("Need at least 1 observation.")
    counts, _ = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    h = -float(np.sum(p * np.log2(p)))
    tic = n * h
    return ESRes(
        measure="total_information_content",
        estimate=tic,
        n=n,
        extra={"entropy_bits": h, "n_bins": bins},
    )


ticmp = total_information_content


def cheatsheet() -> str:
    return "total_information_content(x) -> N * H(X), total information."

# morie.fn — function file (hadesllm/morie)
"""Shannon entropy and related information measures."""

import numpy as np

from ._containers import DescriptiveResult


def entropy(x, base=2, method="histogram", n_bins=20):
    """
    Estimate Shannon entropy H(X) of a continuous or discrete variable.

    :param x: (n,) data.
    :param base: Logarithm base (2=bits, e=nats).
    :param method: 'histogram' or 'plugin' (for discrete data).
    :param n_bins: Bins for histogram method.
    :return: DescriptiveResult with entropy, max possible entropy.

    References
    ----------
    Shannon CE (1948). A Mathematical Theory of Communication. Bell
    System Technical Journal 27(3):379-423.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if method == "plugin":
        _, counts = np.unique(x, return_counts=True)
        probs = counts / n
    else:
        counts, _ = np.histogram(x, bins=n_bins)
        probs = counts / n
        probs = probs[probs > 0]

    h = -np.sum(probs * np.log(probs))
    if base != np.e:
        h /= np.log(base)

    h_max = np.log(len(probs)) / np.log(base) if len(probs) > 0 else 0.0
    redundancy = 1 - h / h_max if h_max > 0 else 0.0

    return DescriptiveResult(
        name="entropy",
        value=float(h),
        extra={
            "entropy": float(h),
            "max_entropy": float(h_max),
            "redundancy": float(redundancy),
            "base": base,
            "method": method,
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "entropy({}) -> Shannon entropy and related information measures."

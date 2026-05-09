"""Zipf's law fit for word frequency distributions."""

import numpy as np

from ._containers import DescriptiveResult


def zipf_law_fit(frequencies):
    """
    Fit Zipf's law to observed frequency data.

    Ranks items by frequency and estimates the Zipf exponent s
    via log-log linear regression: log(freq) ~ -s * log(rank) + c.

    :param frequencies: (n,) observed frequencies.
    :return: DescriptiveResult with Zipf exponent, R-squared.

    References
    ----------
    Zipf GK (1949). Human Behavior and the Principle of Least Effort.
    """
    freq = np.asarray(frequencies, dtype=np.float64).ravel()
    freq = freq[freq > 0]
    ranks = np.arange(1, len(freq) + 1, dtype=np.float64)
    order = np.argsort(-freq)
    freq_sorted = freq[order]

    log_r = np.log(ranks)
    log_f = np.log(freq_sorted)

    A = np.column_stack([log_r, np.ones(len(log_r))])
    coeffs, *_ = np.linalg.lstsq(A, log_f, rcond=None)
    s = -coeffs[0]

    predicted = coeffs[0] * log_r + coeffs[1]
    ss_res = np.sum((log_f - predicted) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return DescriptiveResult(
        name="zipf_law_fit",
        value=float(s),
        extra={
            "zipf_exponent": float(s),
            "intercept": float(coeffs[1]),
            "r_squared": float(r_squared),
            "n_items": len(freq_sorted),
            "max_frequency": float(freq_sorted[0]),
        },
    )


def cheatsheet() -> str:
    return "zipf_law_fit({}) -> Zipf's law fit for word frequency distributions."

# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""A-M bootstrap confidence intervals."""

from __future__ import annotations

from ._containers import DescriptiveResult


def am_confidence_intervals(boot_positions, alpha: float = 0.05) -> DescriptiveResult:
    """Compute percentile confidence intervals from bootstrap A-M positions.

    :param boot_positions: (n_boot x n_stimuli) bootstrap position matrix.
    :param alpha: Significance level (default 0.05 for 95% CI).
    :return: DescriptiveResult with lower and upper bounds per stimulus.

    .. epigraph:: There is no royal road to geometry. -- Euclid
    """
    import numpy as np

    B = np.asarray(boot_positions, dtype=float)
    lo = np.percentile(B, 100 * alpha / 2, axis=0)
    hi = np.percentile(B, 100 * (1 - alpha / 2), axis=0)
    means = B.mean(axis=0)
    return DescriptiveResult(
        name="am_confidence_intervals",
        value=float(1 - alpha),
        extra={
            "lower": lo.tolist(),
            "upper": hi.tolist(),
            "means": means.tolist(),
            "n_boot": B.shape[0],
            "alpha": alpha,
        },
    )


amci = am_confidence_intervals


def cheatsheet() -> str:
    return "am_confidence_intervals({}) -> A-M bootstrap confidence intervals."

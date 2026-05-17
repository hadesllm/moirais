# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Blackbox total sum of squares."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bb_sum_squares(Z) -> DescriptiveResult:
    """Grand-mean total sum of squares for Blackbox fit calculation.

    :param Z: Respondent x issue data matrix.
    :return: DescriptiveResult with total SS.

    .. epigraph:: It is not what happens to you, but how you react, that matters. -- Epictetus
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    grand_mean = np.nanmean(Z)
    ss_total = float(np.nansum((Z - grand_mean) ** 2))
    return DescriptiveResult(
        name="bb_sum_squares",
        value=ss_total,
        extra={"grand_mean": float(grand_mean), "n_obs": int(np.sum(~np.isnan(Z)))},
    )


bbss = bb_sum_squares


def cheatsheet() -> str:
    return "bb_sum_squares({}) -> Blackbox total sum of squares."

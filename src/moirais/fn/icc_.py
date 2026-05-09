# moirais.fn — function file (hadesllm/moirais)
"""Intraclass correlation coefficient (ICC)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def intraclass_correlation(ratings: np.ndarray) -> DescriptiveResult:
    """ICC(1,1), ICC(2,1), ICC(3,1) from a subjects-by-raters matrix.

    Uses one-way and two-way ANOVA mean squares.

    Parameters
    ----------
    ratings : (n_subjects, k_raters) array

    Returns
    -------
    DescriptiveResult
    """
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 2 or ratings.shape[0] < 2 or ratings.shape[1] < 2:
        raise ValueError("Need (n>=2, k>=2) ratings matrix.")

    n, k = ratings.shape
    grand = ratings.mean()

    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)

    ms_r = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_c = n * np.sum((col_means - grand) ** 2) / (k - 1)
    ss_total = np.sum((ratings - grand) ** 2)
    ms_e = (ss_total - (n - 1) * ms_r - (k - 1) * ms_c) / ((n - 1) * (k - 1))
    ms_e = max(ms_e, 1e-12)

    ms_w = (ss_total - (n - 1) * ms_r) / (n * (k - 1))
    ms_w = max(ms_w, 1e-12)

    icc1 = (ms_r - ms_w) / (ms_r + (k - 1) * ms_w)
    icc2 = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
    icc3 = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)

    return DescriptiveResult(
        name="icc",
        value=float(icc3),
        extra={
            "icc_1_1": float(icc1),
            "icc_2_1": float(icc2),
            "icc_3_1": float(icc3),
            "ms_between": float(ms_r),
            "ms_within": float(ms_w),
            "ms_error": float(ms_e),
            "n": n,
            "k": k,
        },
    )


icc_ = intraclass_correlation


def cheatsheet() -> str:
    return "intraclass_correlation({}) -> Intraclass correlation coefficient (ICC)."

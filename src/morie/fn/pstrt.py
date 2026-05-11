# morie.fn — function file (hadesllm/morie)
"""Propensity score stratification. 'You must unlearn what you have learned.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ps_stratify(
    ps: np.ndarray,
    n_strata: int = 5,
) -> DescriptiveResult:
    """
    Stratify observations by propensity score quantiles.

    Divides the sample into *n_strata* equal-sized bins based on the
    estimated propensity score, following Rosenbaum & Rubin (1984).

    :param ps: Estimated propensity scores, array of shape (n,).
    :param n_strata: Number of strata (quintiles = 5). Default 5.
    :return: DescriptiveResult with stratum labels and boundaries.
    :raises ValueError: If ps has wrong shape or n_strata < 2.

    References
    ----------
    Rosenbaum, P. R., & Rubin, D. B. (1984). Reducing bias in
    observational studies using subclassification on the propensity
    score. Journal of the American Statistical Association, 79(387),
    516--524. doi:10.1080/01621459.1984.10478078
    """
    ps = np.asarray(ps, dtype=float)
    if ps.ndim != 1 or ps.size < n_strata:
        raise ValueError(f"ps must be 1-D with length >= n_strata ({n_strata}).")
    if n_strata < 2:
        raise ValueError(f"n_strata must be >= 2, got {n_strata}.")

    quantiles = np.linspace(0, 100, n_strata + 1)
    boundaries = np.percentile(ps, quantiles)
    labels = np.digitize(ps, boundaries[1:-1], right=False)

    unique, counts = np.unique(labels, return_counts=True)
    strata_means = [float(np.mean(ps[labels == s])) for s in unique]

    return DescriptiveResult(
        name="PS Stratification",
        value=n_strata,
        extra={
            "labels": labels,
            "boundaries": boundaries.tolist(),
            "n_strata": n_strata,
            "strata_sizes": dict(zip(unique.tolist(), counts.tolist())),
            "strata_mean_ps": dict(zip(unique.tolist(), strata_means)),
            "n": len(ps),
        },
    )


pstrt = ps_stratify


def cheatsheet() -> str:
    return "ps_stratify({}) -> Propensity score stratification. 'You must unlearn what you "

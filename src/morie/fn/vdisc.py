"""Discriminant validity via HTMT ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult


def discriminant_validity(
    correlation_matrix: np.ndarray,
    subscale_assignments: dict[str, list[int]],
    *,
    threshold: float = 0.85,
) -> DescriptiveResult:
    """Discriminant validity via Heterotrait-Monotrait (HTMT) ratio.

    HTMT compares between-construct correlations to within-construct
    correlations. Values below threshold indicate discriminant validity.

    Parameters
    ----------
    correlation_matrix : ndarray
        Correlation matrix (p x p).
    subscale_assignments : dict
        {factor_name: [item_indices]}.
    threshold : float
        HTMT threshold (default 0.85; strict: 0.90).

    Returns
    -------
    DescriptiveResult
        value=DataFrame with factor pairs and HTMT values.

    References
    ----------
    Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion
    for assessing discriminant validity in variance-based SEM. Journal
    of the Academy of Marketing Science, 43(1), 115-135.
    """
    R = np.asarray(correlation_matrix, dtype=np.float64)
    factors = list(subscale_assignments.keys())

    def mean_het(idx_a, idx_b):
        vals = []
        for i in idx_a:
            for j in idx_b:
                vals.append(abs(R[i, j]))
        return np.mean(vals) if vals else 0.0

    def mean_mono(idx):
        vals = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                vals.append(abs(R[idx[i], idx[j]]))
        return np.mean(vals) if vals else 1.0

    rows = []
    all_pass = True
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            idx_a = subscale_assignments[factors[i]]
            idx_b = subscale_assignments[factors[j]]
            het = mean_het(idx_a, idx_b)
            mono_a = mean_mono(idx_a)
            mono_b = mean_mono(idx_b)
            denom = np.sqrt(max(mono_a * mono_b, 1e-10))
            htmt = het / denom
            passes = htmt < threshold
            if not passes:
                all_pass = False
            rows.append(
                {
                    "factor_1": factors[i],
                    "factor_2": factors[j],
                    "HTMT": float(htmt),
                    "pass": passes,
                }
            )

    return DescriptiveResult(
        name="Discriminant validity (HTMT)",
        value=pd.DataFrame(rows),
        extra={"threshold": threshold, "all_pass": all_pass},
    )


htmt = discriminant_validity


def cheatsheet() -> str:
    return "discriminant_validity({}) -> Discriminant validity via HTMT ratio."

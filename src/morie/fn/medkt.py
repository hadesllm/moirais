# morie.fn — function file (hadesllm/morie)
"""Median test for k independent samples."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["medkt"]


def medkt(*samples, axis=0):
    r"""
    Median test for k≥2 independent samples.

    Generalization of two-sample median test using pooled median.
    Tests H0: all samples have same median.
    """
    if len(samples) < 2:
        raise ValueError("At least 2 samples required")

    samples = [np.asarray(s, dtype=np.float64).ravel() for s in samples]

    # Pooled median
    combined = np.concatenate(samples)
    med_pooled = np.median(combined)

    # Contingency table: k samples × 2 (above/below median)
    k = len(samples)
    contingency = np.zeros((k, 2), dtype=int)

    for i, sample in enumerate(samples):
        contingency[i, 0] = np.sum(sample > med_pooled)
        contingency[i, 1] = np.sum(sample <= med_pooled)

    # Chi-square test
    chi2, p_value, dof, expected_freq = sp_stats.chi2_contingency(contingency)

    return {
        "statistic": float(chi2),
        "p_value": float(p_value),
        "k": int(k),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }

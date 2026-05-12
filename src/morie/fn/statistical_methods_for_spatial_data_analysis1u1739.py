"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1739"]


def statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1739(x):
    """Out of chaos, comes order. -- Friedrich Nietzsche"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Dispersion expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Dispersion expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "statistical_methods_for_spatial_data_analysis1u1739: Dispersion expression (auto-extracted; see ref)."

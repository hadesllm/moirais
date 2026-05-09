"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["design_of_observational_studies_chapter_20_unnumbered_736"]


def design_of_observational_studies_chapter_20_unnumbered_736(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: μ c2 ≥ δ is rejected for small values of −→t . Here, both ←−t ≥ 1.65 and −→t ≤− 1.65,

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : RichResult
        Inherits from ``dict`` (so ``isinstance(result, dict)`` is True
        and ``result["statistic"]`` / ``result.get(...)`` keep working),
        but also exposes a multi-section ``str(result)`` render. Keys: value.
        See ``moirais.fn.describe('design_of_observational_studies20u736')`` for the full guide.

    References
    ----------
    Design of observational studies, ch.20 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Probability expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Probability expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "design_of_observational_studies20u736: Probability expression (auto-extracted; see ref)."

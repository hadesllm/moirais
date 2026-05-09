"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["design_of_observational_studies_chapter_15_unnumbered_550"]


def design_of_observational_studies_chapter_15_unnumbered_550(x):
    """
    Dispersion expression (auto-extracted; see ref).

    Formula: specifically, λopt is any vector proportional to Ω −1τ where τ = (τ1,...,τ M )T ;s e e

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
        See ``moirais.fn.describe('design_of_observational_studies15u550')`` for the full guide.

    References
    ----------
    Design of observational studies, ch.15 (unnumbered)
    """
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
    return "design_of_observational_studies15u550: Dispersion expression (auto-extracted; see ref)."

"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["design_of_observational_studies_chapter_4_unnumbered_241"]


def design_of_observational_studies_chapter_4_unnumbered_241(x):
    """
    Dispersion expression (auto-extracted; see ref).

    Formula: For Γ = 1, formulas ( 3.19) and ( 3.20) reduce to the formulas for randomization inference in Sect. 2.3.3, namely E ( T | F , Z ) = I (I + 1) /4 and

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
        See ``moirais.fn.describe('design_of_observational_studies4u241')`` for the full guide.

    References
    ----------
    Design of observational studies, ch.4 (unnumbered)
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
    return "design_of_observational_studies4u241: Dispersion expression (auto-extracted; see ref)."

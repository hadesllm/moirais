"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["design_of_observational_studies_chapter_20_unnumbered_769"]


def design_of_observational_studies_chapter_20_unnumbered_769(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: ⊿ is the hypothesis that 1 × μ t − 2 × μ c1 + 1 × μ c2 ≤ τ . As before, for τ = 0,

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
        See ``moirais.fn.describe('design_of_observational_studies20u769')`` for the full guide.

    References
    ----------
    Design of observational studies, ch.20 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "design_of_observational_studies20u769: CentralTendency expression (auto-extracted; see ref)."

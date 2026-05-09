"""Logistic equation extracted from Design of observational studies.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["design_of_observational_studies_chapter_9_equation_1"]


def design_of_observational_studies_chapter_9_equation_1(x):
    """
    Logistic equation extracted from Design of observational studies.

    Formula: = ζ0 + ζ1 xℓ1 + ζ2 xℓ2 + ζ3 xℓ3, (9.1)

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
        See ``moirais.fn.describe('design_of_observational_studies9e1')`` for the full guide.

    References
    ----------
    Design of observational studies, ch.9 eq.9.1
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Logistic equation extracted from Design of observational studies.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Logistic equation extracted from Design of observational studies."},
    )


def cheatsheet():
    return "design_of_observational_studies9e1: Logistic equation extracted from Design of observational studies."

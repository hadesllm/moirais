"""Regression equation extracted from Cahoon, Hinich & Ordeshook (1978) Analyzing Spatial Models of Choice and Judgment.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["analyzing_spatial_models_of_choice_and_judgment_chapter_3_equation_34"]


def analyzing_spatial_models_of_choice_and_judgment_chapter_3_equation_34(x):
    """
    Regression equation extracted from Cahoon, Hinich & Ordeshook (1978) Analyzing Spatial Models of Choice and Judgment.

    Formula: Ψ∗∗ = Z

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
        See ``moirais.fn.describe('analyzing_spatial_models_of_choice_and_judgment3e34')`` for the full guide.

    References
    ----------
    Cahoon, Hinich & Ordeshook (1978) Analyzing Spatial Models of Choice and Judgment, ch.3 eq.3.34
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Regression equation extracted from Cahoon, Hinich & Ordeshook (1978) Analyzing Spatial Models of Choice and Judgment.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Regression equation extracted from Cahoon, Hinich & Ordeshook (1978) Analyzing Spatial Models of Choice and Judgment."},
    )


def cheatsheet():
    return "analyzing_spatial_models_of_choice_and_judgment3e34: Regression equation extracted from Cahoon, Hinich & Ordeshook (1978) Analyzing Spatial Models of Choice and Judgment."

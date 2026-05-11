"""CountModels expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_280"]


def analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_280(x):
    """
    CountModels expression (auto-extracted; see ref).

    Formula: p(y jk | α j , β j , ψk , xi ) = Poisson(λ jk )                   (6.48)

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
        See ``morie.fn.describe('analyzing_spatial_models_of_choice_and_judgment5u280')`` for the full guide.

    References
    ----------
    Cahoon, Hinich & Ordeshook (1978) Analyzing Spatial Models of Choice and Judgment, ch.5 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CountModels expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CountModels expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "analyzing_spatial_models_of_choice_and_judgment5u280: CountModels expression (auto-extracted; see ref)."

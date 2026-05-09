"""CausalInference expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["design_of_observational_studies_chapter_2_unnumbered_14"]


def design_of_observational_studies_chapter_2_unnumbered_14(x):
    """
    CausalInference expression (auto-extracted; see ref).

    Formula: had no effect, so that rTi j = rCi j for all i, j, then Yi = (Zi1 − Zi2)( Ri1 − Ri2) is

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
        See ``moirais.fn.describe('design_of_observational_studies2u14')`` for the full guide.

    References
    ----------
    Design of observational studies, ch.2 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CausalInference expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CausalInference expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "design_of_observational_studies2u14: CausalInference expression (auto-extracted; see ref)."

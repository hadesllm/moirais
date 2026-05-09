"""PowerAndDesign expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_1_unnumbered_69"]


def shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_1_unnumbered_69(x):
    """
    PowerAndDesign expression (auto-extracted; see ref).

    Formula: [EQ] 1, 2, · · · , µ({0}) = 1/2 and µ(∅) = 0 and (iii) µ({x}) = 1/2x+1, x ∈ Ω.

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
        See ``moirais.fn.describe('shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a1u69')`` for the full guide.

    References
    ----------
    Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R, ch.1 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="PowerAndDesign expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "PowerAndDesign expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a1u69: PowerAndDesign expression (auto-extracted; see ref)."

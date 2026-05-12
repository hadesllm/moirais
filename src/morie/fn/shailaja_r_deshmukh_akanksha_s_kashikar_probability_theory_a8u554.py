"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_8_unnumbered_554"]


def shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_8_unnumbered_554(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: n = n1+δ, δ > 0, then bn > 0, b n -> ∞ and

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
        See ``morie.fn.describe('shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a8u554')`` for the full guide.

    References
    ----------
    Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R, ch.8 (unnumbered)
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
    return "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a8u554: Probability expression (auto-extracted; see ref)."

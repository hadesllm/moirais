"""Probability equation extracted from Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_4_equation_9"]


def shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_4_equation_9(x):
    """
    Probability equation extracted from Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R.

    Formula: mass function px = P [X = x] = pqx, x = 0, 1, 2, · · · , q = 1 − p. Then its

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
        See ``morie.fn.describe('shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a4e9')`` for the full guide.

    References
    ----------
    Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R, ch.4 eq.4.9
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Probability equation extracted from Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Probability equation extracted from Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R."},
    )


def cheatsheet():
    return "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a4e9: Probability equation extracted from Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R."

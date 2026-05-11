"""CentralTendency equation extracted from Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_10_equation_3"]


def shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_10_equation_3(x):
    """
    CentralTendency equation extracted from Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R.

    Formula: X n ≤ x] → 0 if x < µ , P [X n ≤ x] → 1 if x > µ , as shown in Equation

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
        See ``morie.fn.describe('shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a10e3')`` for the full guide.

    References
    ----------
    Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R, ch.10 eq.10.3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency equation extracted from Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency equation extracted from Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R."},
    )


def cheatsheet():
    return "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a10e3: CentralTendency equation extracted from Shailaja R. Deshmukh, Akanksha S. Kashikar - Probability Theory  An Introduction Using R."

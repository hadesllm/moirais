"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon_chapter_7_unnumbered_25"]


def jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon_chapter_7_unnumbered_25(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: if a =b and b =c, then of coursea =c (again, that’s just saying that ifa and b are

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
        See ``morie.fn.describe('jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon7u25')`` for the full guide.

    References
    ----------
    Jay Cummings - Proofs  A Long-Form Mathematics Textbook (The Long-Form Math Textbook Series)-Independently published (2021), ch.7 (unnumbered)
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
    return "jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon7u25: CentralTendency expression (auto-extracted; see ref)."

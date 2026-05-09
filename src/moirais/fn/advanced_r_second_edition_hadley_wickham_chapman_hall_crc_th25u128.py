"""Association expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_128"]


def advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_128(x):
    """
    Association expression (auto-extracted; see ref).

    Formula: [EQ] for(int i = 0; i < N; i++) {

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
        See ``moirais.fn.describe('advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u128')`` for the full guide.

    References
    ----------
    Advanced R (Second Edition) -- Hadley Wickham -- Chapman & Hall CRC the R, ch.25 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return hypothesis_test_result(
            test_name="Association expression (auto-extracted; see ref).",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Association expression (auto-extracted; see ref)."},
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Association expression (auto-extracted; see ref).",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "Association expression (auto-extracted; see ref).", "p_value": float(result.pvalue)},
    )


def cheatsheet():
    return "advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u128: Association expression (auto-extracted; see ref)."

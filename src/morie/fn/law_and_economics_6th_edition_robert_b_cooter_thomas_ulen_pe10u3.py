"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe_chapter_10_unnumbered_3"]


def law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe_chapter_10_unnumbered_3(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: j = 100..513j - 152 + .51-152 = 0.

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
        See ``morie.fn.describe('law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe10u3')`` for the full guide.

    References
    ----------
    Law and Economics, 6th Edition -- Robert B. Cooter, Thomas Ulen -- Pearson Series in Economics, 6th Edition, 2011 -- Prentice Hall -- 9780132540650 -- 4bc55da23884b7603280fecc527f4743 -- Anna’s Archive, ch.10 (unnumbered)
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
    return "law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe10u3: Probability expression (auto-extracted; see ref)."

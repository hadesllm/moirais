"""Regression expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_22"]


def springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_22(x):
    """
    Regression expression (auto-extracted; see ref).

    Formula: instance, thez-statistic associated withβ1 is equal toˆβ1/SE(ˆβ1), and so a

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
        See ``morie.fn.describe('springer_texts_in_statistics_series_gareth_james_daniela_wit2u22')`` for the full guide.

    References
    ----------
    [Springer Texts in Statistics Series] Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani - An Introduction To Statistical Learning  With Applications In R, ch.2 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Regression expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Regression expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "springer_texts_in_statistics_series_gareth_james_daniela_wit2u22: Regression expression (auto-extracted; see ref)."

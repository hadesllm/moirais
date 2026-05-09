"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_3_unnumbered_100"]


def springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_3_unnumbered_100(x):
    """
    Dispersion expression (auto-extracted; see ref).

    Formula: This gives us a very good idea of the accuracy of ˆα:S E ( ˆα) ≈0.083. So

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
        See ``moirais.fn.describe('springer_texts_in_statistics_series_gareth_james_daniela_wit3u100')`` for the full guide.

    References
    ----------
    [Springer Texts in Statistics Series] Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani - An Introduction To Statistical Learning  With Applications In R, ch.3 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Dispersion expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Dispersion expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "springer_texts_in_statistics_series_gareth_james_daniela_wit3u100: Dispersion expression (auto-extracted; see ref)."

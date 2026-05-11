"""PowerAndDesign expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_284"]


def chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_284(x):
    """
    PowerAndDesign expression (auto-extracted; see ref).

    Formula: if Yi = τ + εi where the εi are sampled from the standard Normal distribution and τ = 1/2, then

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
        See ``morie.fn.describe('chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u284')`` for the full guide.

    References
    ----------
    [Chapman & Hall CRC Handbooks of Modern Statistical Methods] José R. Zubizarreta, Elizabeth A. Stuart, Dylan S. Small, Paul R - Handbook of Matching and Weighting Adjustments for Causal Inference, ch.10 (unnumbered)
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
    return "chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u284: PowerAndDesign expression (auto-extracted; see ref)."

"""GeneralStatistics expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_10_unnumbered_16"]


def statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_10_unnumbered_16(x, cdf=None):
    """
    GeneralStatistics expression (auto-extracted; see ref).

    Formula: [EQ] 50.9% of 1.6% = 0.509 × 1.6% ≈ 0.8o f1 %

    Parameters
    ----------
    x : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : RichResult
        Inherits from ``dict`` (so ``isinstance(result, dict)`` is True
        and ``result["statistic"]`` / ``result.get(...)`` keep working),
        but also exposes a multi-section ``str(result)`` render. Keys: value.
        See ``moirais.fn.describe('statistics_fourth_edition_david_freeman_robert_pisani_and_ro10u16')`` for the full guide.

    References
    ----------
    Statistics, Fourth Edition -- David Freeman, Robert Pisani, and Roger Purves -- 2018, ch.10 (unnumbered)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return hypothesis_test_result(
            test_name="GeneralStatistics expression (auto-extracted; see ref).",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<2: insufficient data."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "GeneralStatistics expression (auto-extracted; see ref).", "p_value": float("nan")},
        )
    x_sorted = np.sort(x)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(x), scale=np.std(x, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return hypothesis_test_result(
        test_name="GeneralStatistics expression (auto-extracted; see ref).",
        statistic=float(statistic),
        pvalue=float(p_value),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "GeneralStatistics expression (auto-extracted; see ref).", "p_value": float(p_value)},
    )


def cheatsheet():
    return "statistics_fourth_edition_david_freeman_robert_pisani_and_ro10u16: GeneralStatistics expression (auto-extracted; see ref)."

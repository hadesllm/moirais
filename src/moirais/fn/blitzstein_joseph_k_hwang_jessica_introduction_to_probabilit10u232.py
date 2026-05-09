"""GeneralStatistics expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_232"]


def blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_232(x, cdf=None):
    """Study the past if you would define the future. — Confucius"""
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
    return "blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u232: GeneralStatistics expression (auto-extracted; see ref)."

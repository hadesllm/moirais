"""GeneralStatistics equation extracted from spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_equation_1"]


def spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_equation_1(x, cdf=None):
    """
    GeneralStatistics equation extracted from spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818.

    Formula: [EQ] where θ = pr( ¯C)/pr(C), 0 < θ < ∞ (Zidek, 1984)

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
        See ``moirais.fn.describe('spatiotemporal_methods_in_environmental_epidemiology_with_r_13e1')`` for the full guide.

    References
    ----------
    spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818, ch.13 eq.13.1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return hypothesis_test_result(
            test_name="GeneralStatistics equation extracted from spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818.",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<2: insufficient data."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "GeneralStatistics equation extracted from spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818.", "p_value": float("nan")},
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
        test_name="GeneralStatistics equation extracted from spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818.",
        statistic=float(statistic),
        pvalue=float(p_value),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "GeneralStatistics equation extracted from spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818.", "p_value": float(p_value)},
    )


def cheatsheet():
    return "spatiotemporal_methods_in_environmental_epidemiology_with_r_13e1: GeneralStatistics equation extracted from spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818."

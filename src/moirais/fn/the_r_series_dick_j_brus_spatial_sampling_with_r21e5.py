"""Correlation equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_5"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_5(x):
    """
    Correlation equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.

    Formula: vgmodel <- vgm(model = "Exp", psill = thetas[

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
        See ``moirais.fn.describe('the_r_series_dick_j_brus_spatial_sampling_with_r21e5')`` for the full guide.

    References
    ----------
    [The R Series] Dick J. Brus - Spatial Sampling with R, ch.21 eq.21.5
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return hypothesis_test_result(
            test_name="Correlation equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Correlation equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."},
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Correlation equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "Correlation equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.", "p_value": float(result.pvalue)},
    )


def cheatsheet():
    return "the_r_series_dick_j_brus_spatial_sampling_with_r21e5: Correlation equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."

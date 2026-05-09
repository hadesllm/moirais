"""GeneralStochastic expression (auto-extracted; see reference).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["stochastic_physics_section_3_unnumbered_10"]


def stochastic_physics_section_3_unnumbered_10(x):
    """
    GeneralStochastic expression (auto-extracted; see reference).

    Formula: victim’s payoff is further decreased to 1−δ−ϵ. Here ϵ may

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
        See ``moirais.fn.describe('sp_s3u10')`` for the full guide.

    References
    ----------
    D'Orsogna & Perc (2015) Statistical physics of crime: A review, §3 (SELF-EXCITING POINT PROCESS MODELING, unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="GeneralStochastic expression (auto-extracted; see reference).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "GeneralStochastic expression (auto-extracted; see reference)."},
    )


def cheatsheet():
    return "sp_s3u10: GeneralStochastic expression (auto-extracted; see reference)."

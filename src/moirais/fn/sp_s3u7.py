"""GeneralStochastic expression (auto-extracted; see reference).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["stochastic_physics_section_3_unnumbered_7"]


def stochastic_physics_section_3_unnumbered_7(x):
    """
    GeneralStochastic expression (auto-extracted; see reference).

    Formula: δ = 0.3,θ = 0.6 andϵ = 0.2, but qualitative results are independent of parameters. This figure has been reproduced from [33].

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
        See ``moirais.fn.describe('sp_s3u7')`` for the full guide.

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
    return "sp_s3u7: GeneralStochastic expression (auto-extracted; see reference)."

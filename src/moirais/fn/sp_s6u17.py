"""GeneralStochastic expression (auto-extracted; see reference).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["stochastic_physics_section_6_unnumbered_17"]


def stochastic_physics_section_6_unnumbered_17(x):
    """
    GeneralStochastic expression (auto-extracted; see reference).

    Formula: τ = 1.5 and θ = 0.35. This indicates that the available resources C need to be balanced so that there is enough stick

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
        See ``moirais.fn.describe('sp_s6u17')`` for the full guide.

    References
    ----------
    D'Orsogna & Perc (2015) Statistical physics of crime: A review, §6 (REHABILITA TION AND RECIDIVISM, unnumbered)
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
    return "sp_s6u17: GeneralStochastic expression (auto-extracted; see reference)."

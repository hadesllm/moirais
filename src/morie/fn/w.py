"""Cohen's w effect size for chi-squared."""

import math
from typing import Union

import numpy as np

from ._containers import ESRes


def cohens_w(
    observed: Union[np.ndarray, list],
    expected: Union[np.ndarray, list] | None = None,
) -> ESRes:
    """Cohen's w for chi-squared.

    w = sqrt(chi2 / N)

    Parameters
    ----------
    observed : array-like
        Observed frequencies.
    expected : array-like or None
        Expected frequencies (uniform if None).

    Returns
    -------
    ESRes
    """
    obs = np.asarray(observed, dtype=np.float64)
    if expected is None:
        exp = np.full_like(obs, obs.sum() / len(obs))
    else:
        exp = np.asarray(expected, dtype=np.float64)
    n = obs.sum()
    chi2 = np.sum((obs - exp) ** 2 / (exp + 1e-15))
    w_val = math.sqrt(chi2 / n) if n > 0 else 0.0
    return ESRes(
        measure="Cohen's w",
        estimate=float(w_val),
        n=int(n),
    )


cw = cohens_w


def cheatsheet() -> str:
    return "cohens_w({}) -> Cohen's w effect size for chi-squared."

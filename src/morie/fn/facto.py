# morie.fn — function file (hadesllm/morie)
"""Full factorial design matrix."""

from __future__ import annotations

from itertools import product as _product

import numpy as np

from ._containers import DescriptiveResult


def factorial_design(
    n_factors: int = 3,
    levels: int = 2,
) -> DescriptiveResult:
    """Generate a full factorial design matrix.

    For *k* factors each at *m* levels, produces :math:`m^k` runs.
    Levels are coded as integers 0, 1, ..., m-1 (or -1, +1 for 2
    levels).

    Parameters
    ----------
    n_factors : int, default 3
        Number of factors (k >= 1).
    levels : int, default 2
        Number of levels per factor (m >= 2).

    Returns
    -------
    DescriptiveResult
        ``value`` is the design matrix (n_runs x n_factors).
        ``extra`` has ``n_runs``, ``n_factors``, ``levels``.

    References
    ----------
    Montgomery, D. C. (2017). *Design and Analysis of Experiments*
    (9th ed.). Wiley. Ch. 5--6.
    """
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors}.")
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}.")

    if levels == 2:
        level_vals = [-1, 1]
    else:
        level_vals = list(range(levels))

    runs = list(_product(level_vals, repeat=n_factors))
    design = np.array(runs, dtype=np.float64)

    return DescriptiveResult(
        name="FactorialDesign",
        value=design,
        extra={
            "n_runs": len(runs),
            "n_factors": n_factors,
            "levels": levels,
        },
    )


facto = factorial_design


def cheatsheet() -> str:
    return "factorial_design({}) -> Full factorial design matrix."

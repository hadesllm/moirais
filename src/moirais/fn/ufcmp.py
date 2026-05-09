"""Compare unfolding methods. 'Kamehameha!' -- Goku, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def compare_unfolding_methods(result1, result2):
    """Compare two unfolding results by stress and correlation.

    Parameters
    ----------
    result1 : dict
        Must have 'stress' and 'coords' keys.
    result2 : dict
        Must have 'stress' and 'coords' keys.

    Returns
    -------
    DescriptiveResult
        value = dict with comparison metrics.
    """
    import numpy as np

    s1 = float(result1["stress"])
    s2 = float(result2["stress"])
    c1 = np.asarray(result1["coords"], dtype=float).ravel()
    c2 = np.asarray(result2["coords"], dtype=float).ravel()
    min_len = min(len(c1), len(c2))
    r = float(np.corrcoef(c1[:min_len], c2[:min_len])[0, 1]) if min_len > 1 else 0.0

    comparison = {
        "stress_1": s1,
        "stress_2": s2,
        "stress_diff": s1 - s2,
        "better": 1 if s1 < s2 else 2,
        "coord_correlation": r,
    }
    return DescriptiveResult(name="compare_unfolding_methods", value=comparison, extra={})


ufcmp = compare_unfolding_methods


def cheatsheet() -> str:
    return "compare_unfolding_methods({}) -> Compare unfolding methods. 'Kamehameha!' -- Goku, Dragon Bal"

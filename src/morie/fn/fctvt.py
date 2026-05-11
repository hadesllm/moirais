# morie.fn — function file (hadesllm/morie)
"""Factor/convert ordinal response levels."""

from __future__ import annotations

from ._containers import DescriptiveResult


def factor_convert_levels(data, levels=None) -> DescriptiveResult:
    """Convert response data to integer factor codes.

    :param data: Raw response array.
    :param levels: Ordered list of valid levels (default: unique sorted values).
    :return: DescriptiveResult with integer-coded data.

    .. epigraph:: "Getsuga Tensho!" -- Ichigo Kurosaki, Bleach
    """
    import numpy as np

    X = np.asarray(data).ravel()
    if levels is None:
        levels = sorted(set(X[~np.equal(X, None)]))
    level_map = {v: i for i, v in enumerate(levels)}
    coded = np.array([level_map.get(x, -1) for x in X], dtype=int)
    return DescriptiveResult(
        name="factor_convert_levels",
        value=len(levels),
        extra={"coded": coded.tolist(), "levels": [str(l) for l in levels], "n_missing": int(np.sum(coded == -1))},
    )


fctvt = factor_convert_levels


def cheatsheet() -> str:
    return "factor_convert_levels({}) -> Factor/convert ordinal response levels."

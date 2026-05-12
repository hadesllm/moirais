# morie.fn -- function file (hadesllm/morie)
"""Recode responses: replace missing/special codes with NaN."""

from __future__ import annotations

from ._containers import DescriptiveResult


def recode_responses(data, missing_codes=None) -> DescriptiveResult:
    """Replace special missing codes with NaN in survey response data.

    :param data: Response matrix.
    :param missing_codes: List of codes to treat as missing (default [77, 88, 99]).
    :return: DescriptiveResult with cleaned data.

    .. epigraph:: "Sasageyo!" -- Survey Corps, Attack on Titan
    """
    import numpy as np

    if missing_codes is None:
        missing_codes = [77, 88, 99]
    X = np.asarray(data, dtype=float).copy()
    n_replaced = 0
    for code in missing_codes:
        mask = code == X
        n_replaced += int(mask.sum())
        X[mask] = np.nan
    return DescriptiveResult(
        name="recode_responses",
        value=n_replaced,
        extra={"cleaned": X.tolist(), "n_replaced": n_replaced, "missing_codes": missing_codes},
    )


rcode = recode_responses


def cheatsheet() -> str:
    return "recode_responses({}) -> Recode responses: replace missing/special codes with NaN."

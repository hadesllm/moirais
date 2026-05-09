# moirais.fn — function file (hadesllm/moirais)
"""Mental health comorbidity index."""

import numpy as np

from ._containers import DescriptiveResult


def comorbidity_index(
    diagnosis_flags: list | np.ndarray,
) -> DescriptiveResult:
    """Count mental health comorbidities from binary diagnosis flags.

    Parameters
    ----------
    diagnosis_flags : array-like
        Binary flags (0/1) for each disorder. Can be 1-D (one person)
        or 2-D (rows=persons, cols=disorders).

    Returns
    -------
    DescriptiveResult
    """
    a = np.asarray(diagnosis_flags, dtype=int)
    if a.ndim == 1:
        total = int(np.sum(a))
        return DescriptiveResult(
            name="comorbidity_index",
            value=float(total),
            extra={"n_disorders_assessed": len(a)},
        )

    counts = a.sum(axis=1)
    return DescriptiveResult(
        name="comorbidity_index",
        value=float(np.mean(counts)),
        extra={
            "median": float(np.median(counts)),
            "pct_0": float(np.mean(counts == 0) * 100),
            "pct_1": float(np.mean(counts == 1) * 100),
            "pct_2plus": float(np.mean(counts >= 2) * 100),
            "n": len(counts),
        },
    )


mhcom = comorbidity_index


def cheatsheet() -> str:
    return "comorbidity_index({}) -> Mental health comorbidity index."

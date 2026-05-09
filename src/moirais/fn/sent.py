# moirais.fn — function file (hadesllm/moirais)
"""Sentence length distribution stats. 'Who'Difficulties strengthen the mind, as labor does the body. — Seneca'

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def sentence_stats(lengths: np.ndarray) -> DescriptiveResult:
    """Descriptive statistics for sentence lengths (months).

    Parameters
    ----------
    lengths : ndarray
        Array of sentence lengths.

    Returns
    -------
    DescriptiveResult
    """
    lengths = np.asarray(lengths, dtype=float)
    if len(lengths) == 0:
        raise ValueError("lengths must not be empty")
    q25, median, q75 = np.percentile(lengths, [25, 50, 75])
    return DescriptiveResult(
        name="Sentence statistics",
        value=float(median),
        extra={
            "mean": float(np.mean(lengths)),
            "median": float(median),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(q75 - q25),
            "min": float(np.min(lengths)),
            "max": float(np.max(lengths)),
            "std": float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0,
            "n": len(lengths),
        },
    )


sent = sentence_stats


def cheatsheet() -> str:
    return "sentence_stats({}) -> Sentence length distribution stats. 'Who's the more foolish?"

"""An unexamined life is not worth living. — Socrates"""

from __future__ import annotations

from collections import Counter

import numpy as np

from moirais.fn._containers import CrimeResult


def tps_use_of_force(
    force_types: list[str] | np.ndarray,
    n_encounters: int,
) -> CrimeResult:
    """An unexamined life is not worth living. — Socrates"""
    if n_encounters <= 0:
        raise ValueError("n_encounters must be positive")
    ft = list(force_types)
    n_uof = len(ft)
    rate = n_uof / n_encounters
    counts = dict(Counter(ft))
    return CrimeResult(
        name="use_of_force_rate",
        rate=rate,
        n=n_uof,
        population=n_encounters,
        extra={"type_counts": counts, "n_types": len(counts)},
    )


tpsuof = tps_use_of_force


def cheatsheet() -> str:
    return "An unexamined life is not worth living. — Socrates"

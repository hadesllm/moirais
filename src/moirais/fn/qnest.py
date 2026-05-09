# moirais.fn — function file (hadesllm/moirais)
"""Qn scale estimator. 'Judge me by my size, do you?'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def qn_estimator(x: np.ndarray) -> DescriptiveResult:
    """Qn robust scale estimator (Rousseeuw & Croux, 1993).

    The Qn is the first quartile of all pairwise |x_i - x_j|
    distances, scaled by a consistency factor for Gaussian data.
    It has 50% breakdown point and 82% Gaussian efficiency —
    better than MAD in many settings.

    :param x: 1-D numeric array.
    :return: DescriptiveResult with Qn estimate.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return DescriptiveResult(name="qnest", value=0.0, extra={"n": n})

    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(abs(x[i] - x[j]))
    diffs = np.array(diffs)

    h = n // 2 + 1
    k = h * (h - 1) // 2
    k = min(k, len(diffs)) - 1
    k = max(k, 0)
    qn_raw = float(np.sort(diffs)[k])

    c_n = 2.2219
    if n <= 9:
        small_c = {2: 0.399, 3: 0.994, 4: 0.512, 5: 0.844, 6: 0.611, 7: 0.857, 8: 0.669, 9: 0.872}
        c_n = small_c.get(n, 2.2219)

    qn = c_n * qn_raw

    return DescriptiveResult(
        name="qnest",
        value=qn,
        extra={"n": n, "qn_raw": qn_raw, "correction_factor": c_n},
    )


qnest = qn_estimator


def cheatsheet() -> str:
    return "qn_estimator({}) -> Qn scale estimator. 'Judge me by my size, do you?'"

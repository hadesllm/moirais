# morie.fn — function file (hadesllm/morie)
"""AR model distance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There is another."


def model_distance(ar1, ar2, **kwargs) -> DescriptiveResult:
    """Compute AR model distance via reflection coefficients.

    Converts both AR models to reflection coefficients and computes
    the Euclidean distance between them.

    Parameters
    ----------
    ar1 : array-like
        AR coefficients of model 1 (with leading 1).
    ar2 : array-like
        AR coefficients of model 2 (with leading 1).

    Returns
    -------
    DescriptiveResult
    """
    ar1 = np.asarray(ar1, dtype=float)
    ar2 = np.asarray(ar2, dtype=float)

    def _ar_to_rc(a):
        a = a.copy()
        if abs(a[0]) > 0:
            a = a / a[0]
        a = a[1:]
        p = len(a)
        rc = np.zeros(p)
        for i in range(p - 1, -1, -1):
            rc[i] = a[i]
            if abs(rc[i]) >= 1.0:
                break
            prev = a[:i].copy()
            for j in range(i):
                a[j] = (prev[j] - rc[i] * prev[i - 1 - j]) / (1.0 - rc[i] ** 2)
        return rc

    rc1 = _ar_to_rc(ar1)
    rc2 = _ar_to_rc(ar2)
    maxlen = max(len(rc1), len(rc2))
    rc1 = np.pad(rc1, (0, maxlen - len(rc1)))
    rc2 = np.pad(rc2, (0, maxlen - len(rc2)))
    dist = float(np.sqrt(np.sum((rc1 - rc2) ** 2)))
    return DescriptiveResult(
        name="model_distance",
        value=dist,
        extra={"distance": dist, "rc1": rc1, "rc2": rc2},
    )


mdist = model_distance


def cheatsheet() -> str:
    return "model_distance({}) -> AR model distance."

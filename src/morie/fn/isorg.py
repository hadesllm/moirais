# morie.fn — function file (hadesllm/morie)
"""Isotonic regression. 'One Punch!' -- Saitama, One Punch Man"""

from __future__ import annotations

from ._containers import DescriptiveResult


def isotonic_regression(x, w=None):
    """Pool-adjacent-violators (PAV) isotonic regression.

    Parameters
    ----------
    x : array-like
        Input values.
    w : array-like or None
        Weights. Defaults to uniform.

    Returns
    -------
    DescriptiveResult
        value = monotonically fitted values (ndarray).
    """
    import numpy as np

    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float).copy()

    target = x.copy()
    block_w = w.copy()
    i = 0
    while i < n - 1:
        if target[i] > target[i + 1]:
            pooled = (target[i] * block_w[i] + target[i + 1] * block_w[i + 1]) / (block_w[i] + block_w[i + 1])
            target[i] = pooled
            target[i + 1] = pooled
            block_w[i] = block_w[i] + block_w[i + 1]
            block_w[i + 1] = block_w[i]
            while i > 0 and target[i - 1] > target[i]:
                i -= 1
                pooled = (target[i] * block_w[i] + target[i + 1] * block_w[i + 1]) / (block_w[i] + block_w[i + 1])
                target[i] = pooled
                target[i + 1] = pooled
                block_w[i] = block_w[i] + block_w[i + 1]
                block_w[i + 1] = block_w[i]
        i += 1

    result = np.zeros(n)
    val = target[0]
    start = 0
    for j in range(1, n):
        if target[j] != val:
            result[start:j] = val
            val = target[j]
            start = j
    result[start:n] = val
    return DescriptiveResult(name="isotonic_regression", value=result, extra={"n": n})


isorg = isotonic_regression


def cheatsheet() -> str:
    return "isotonic_regression({}) -> Isotonic regression. 'One Punch!' -- Saitama, One Punch Man"

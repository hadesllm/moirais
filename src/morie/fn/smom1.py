"""Raw moment."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Great, kid. Don't get cocky."


def raw_moment(x, k=1, **kwargs) -> DescriptiveResult:
    """Compute the *k*-th raw moment of signal *x*.

    .. math::

        m_k = \\frac{1}{N} \\sum_{n=0}^{N-1} x^k(n)

    Parameters
    ----------
    x : array-like
        Input signal.
    k : int
        Moment order (default 1).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    mk = float(np.mean(x**k))
    return DescriptiveResult(
        name="raw_moment",
        value=mk,
        extra={"moment_order": k, "raw_moment": mk, "n": len(x)},
    )


smom1 = raw_moment


def cheatsheet() -> str:
    return "raw_moment({}) -> Raw moment."

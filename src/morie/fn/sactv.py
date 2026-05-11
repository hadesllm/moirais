# morie.fn — function file (hadesllm/morie)
"""Hjorth activity parameter."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The man who moves a mountain begins by carrying away small stones. — Confucius"


def activity(x, **kwargs) -> DescriptiveResult:
    """Compute the Hjorth Activity parameter.

    .. math::

        \\text{Activity} = \\text{var}(x)

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    act = float(np.var(x, ddof=0))
    return DescriptiveResult(
        name="activity",
        value=act,
        extra={"activity": act, "n": len(x)},
    )


sactv = activity


def cheatsheet() -> str:
    return "activity({}) -> Hjorth activity parameter."

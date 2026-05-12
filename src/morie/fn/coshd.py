# morie.fn -- function file (hadesllm/morie)
"""Cosh spectral distance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Let go of your hate."


def cosh_distance(S1, S2, **kwargs) -> DescriptiveResult:
    r"""Compute the cosh spectral distance.

    .. math::

        d_{\\cosh} = \\frac{1}{K}\\sum_k \\left(\\cosh\\left(\\ln\\frac{S_1(k)}{S_2(k)}\\right) - 1\\right)

    Parameters
    ----------
    S1 : array-like
        Power spectrum 1.
    S2 : array-like
        Power spectrum 2.

    Returns
    -------
    DescriptiveResult
    """
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)
    S1 = np.maximum(S1, 1e-30)
    S2 = np.maximum(S2, 1e-30)
    log_ratio = np.log(S1 / S2)
    d = float(np.mean(np.cosh(log_ratio) - 1.0))
    return DescriptiveResult(
        name="cosh_distance",
        value=d,
        extra={"cosh_dist": d},
    )


coshd = cosh_distance


def cheatsheet() -> str:
    return "cosh_distance({}) -> Cosh spectral distance."

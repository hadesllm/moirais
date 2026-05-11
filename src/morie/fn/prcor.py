# morie.fn — function file (hadesllm/morie)
"""Partial autocorrelation (PARCOR) coefficients via lattice method."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def parcor_fn(x: np.ndarray, order: int = 10) -> DescriptiveResult:
    """Compute partial autocorrelation coefficients via lattice filter.

    :param x: 1-D input signal.
    :param order: Maximum lag order (default 10).
    :return: DescriptiveResult with parcor coefficients in extra.
    """
    from morie._armodel import parcor_coefficients

    x = np.asarray(x, dtype=float).ravel()
    k = parcor_coefficients(x, order=order)
    return DescriptiveResult(
        name="parcor_coefficients",
        value=None,
        extra={"parcor": k},
    )


prcor = parcor_fn


def cheatsheet() -> str:
    return "parcor_fn({}) -> Partial autocorrelation (PARCOR) coefficients via lattice me"

"""Teager-Kaiser energy operator."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The man who moves a mountain begins by carrying away small stones. -- Confucius"


def teager_energy(x, **kwargs) -> DescriptiveResult:
    r"""Compute Teager-Kaiser energy operator.

    .. math::
        \\Psi[x(n)] = x(n)^2 - x(n-1) \\cdot x(n+1)

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        energy = np.array([])
    else:
        energy = x[1:-1] ** 2 - x[:-2] * x[2:]
    mean_energy = float(np.mean(energy)) if len(energy) > 0 else 0.0
    return DescriptiveResult(
        name="teager_energy",
        value=mean_energy,
        extra={
            "energy": energy,
            "max_energy": float(np.max(energy)) if len(energy) > 0 else 0.0,
        },
    )


tkeo = teager_energy


def cheatsheet() -> str:
    return "teager_energy({}) -> Teager-Kaiser energy operator."

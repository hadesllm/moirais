"""Witten index Tr((-1)^F e^{-beta H})."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def witten_index(
    beta: float = 1.0,
    E_levels: list[float] | np.ndarray | None = None,
    fermion_numbers: list[int] | np.ndarray | None = None,
) -> DescriptiveResult:
    """Compute the Witten index for a supersymmetric system.

    .. math::

        W = \\text{Tr}\\bigl[(-1)^F e^{-\\beta H}\\bigr]

    The Witten index counts n_boson(E=0) - n_fermion(E=0) and is
    topological (beta-independent for exact SUSY).

    :param beta: Inverse temperature. Must be > 0.
    :param E_levels: Energy levels. Defaults to sample spectrum.
    :param fermion_numbers: Fermion number (0=boson, 1=fermion) per level.
    :return: DescriptiveResult with Witten index.
    """
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}.")
    if E_levels is None:
        E_levels = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        fermion_numbers = np.array([0, 1, 0, 1, 0, 1])
    E_levels = np.asarray(E_levels, dtype=float)
    fermion_numbers = np.asarray(fermion_numbers, dtype=int)
    signs = (-1.0) ** fermion_numbers
    W = np.sum(signs * np.exp(-beta * E_levels))
    n_boson_ground = int(np.sum((E_levels == 0) & (fermion_numbers == 0)))
    n_fermion_ground = int(np.sum((E_levels == 0) & (fermion_numbers == 1)))
    return DescriptiveResult(
        name="witten_index",
        value=float(W),
        extra={
            "witten_index": float(W),
            "beta": beta,
            "n_levels": len(E_levels),
            "n_boson_ground": n_boson_ground,
            "n_fermion_ground": n_fermion_ground,
            "susy_broken": abs(W) < 1e-12,
        },
    )


def cheatsheet() -> str:
    return "witten_index(beta, E_levels) -> Witten index Tr((-1)^F e^{-bH})"


witin = witten_index

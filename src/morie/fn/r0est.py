# morie.fn -- function file (hadesllm/morie)
"""R0 estimation via the next-generation matrix method."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def r0_next_generation(
    F: list[list[float]],
    V: list[list[float]],
) -> ESRes:
    """Estimate R0 using the next-generation matrix (NGM) method.

    R0 is the spectral radius (dominant eigenvalue) of the
    next-generation matrix K = F V^{-1}, where F is new infections
    and V is transitions between compartments.

    .. math::

        R_0 = \\rho(F V^{-1})

    Parameters
    ----------
    F : list[list[float]]
        Transmission matrix (new infections).
    V : list[list[float]]
        Transition matrix (compartment changes).

    Returns
    -------
    ESRes

    References
    ----------
    Diekmann, O., Heesterbeek, J. A. P. & Metz, J. A. J. (1990).
    On the definition and the computation of the basic reproduction
    ratio R0 in models for infectious diseases in heterogeneous
    populations. Journal of Mathematical Biology, 28(4), 365-382.

    van den Driessche, P. & Watmough, J. (2002). Reproduction numbers
    and sub-threshold endemic equilibria for compartmental models of
    disease transmission. Mathematical Biosciences, 180(1-2), 29-48.
    """
    F_arr = np.array(F, dtype=float)
    V_arr = np.array(V, dtype=float)

    if F_arr.shape != V_arr.shape:
        raise ValueError("F and V must have the same shape")
    if F_arr.ndim != 2 or F_arr.shape[0] != F_arr.shape[1]:
        raise ValueError("F and V must be square matrices")

    det_V = np.linalg.det(V_arr)
    if abs(det_V) < 1e-15:
        raise ValueError("V must be invertible")

    K = F_arr @ np.linalg.inv(V_arr)
    eigvals = np.linalg.eigvals(K)
    r0_val = float(np.max(np.abs(eigvals)))

    return ESRes(
        measure="R0_NGM",
        estimate=r0_val,
        extra={
            "method": "next_generation_matrix",
            "K_spectral_radius": r0_val,
            "eigenvalues": eigvals.tolist(),
        },
    )


r0est = r0_next_generation


def cheatsheet() -> str:
    return "r0_next_generation({}) -> R0 via next-generation matrix method."
